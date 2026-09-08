"""The two classifiers that stand between an autonomous agent and your card.

Both are deliberately biased toward over-reporting. A false positive costs one
confirmation; a false negative costs money, or a password.
"""

from __future__ import annotations

import pytest

from roost.agent.tools.browser import classify_click, classify_field
from roost.protocol.agent import Risk


def el(**kw):
    base = {'ref': 1, 'tag': 'button', 'type': '', 'name': '', 'id': '', 'label': '', 'value': '', 'href': ''}
    base.update(kw)
    return base


BUYING = [
    'Place your order', 'Place Your Order', 'Buy now', 'Buy it now',
    'Complete purchase', 'Confirm and pay', 'Confirm order', 'Pay now',
    'Submit order', 'Proceed to payment', 'Subscribe', 'Start my free trial',
    'Donate', 'Place bid', 'Book now', 'Reserve now', 'Checkout', 'Check out',
    'Pay', 'Authorize payment',
]


@pytest.mark.parametrize('label', BUYING)
def test_buying_buttons_are_purchases(label):
    risk, _ = classify_click(el(label=label))
    assert risk is Risk.PURCHASE, label


def test_a_purchase_word_anywhere_in_the_element_counts():
    """The visible label is not the only place it shows up."""
    assert classify_click(el(label='', name='placeOrderBtn'))[0] is Risk.PURCHASE
    assert classify_click(el(label='', href='/checkout/confirm'))[0] is Risk.PURCHASE


ORDINARY = [
    'Add to wishlist', 'Search', 'Next page', 'Sign in', 'Read more',
    'Filter results', 'Sort by price', 'Compare', 'Back', 'Close',
]


@pytest.mark.parametrize('label', ORDINARY)
def test_ordinary_buttons_are_not_purchases(label):
    assert classify_click(el(label=label))[0] is not Risk.PURCHASE, label


def test_adding_to_a_cart_is_flagged_but_is_not_a_purchase():
    """It is the step before buying, not buying."""
    risk, why = classify_click(el(label='Add to cart'))
    assert risk is Risk.WRITE and 'cart' in why


SECRET_FIELDS = [
    {'type': 'password'},
    {'name': 'password'}, {'name': 'passwd'}, {'name': 'user_pwd'},
    {'name': 'cardNumber'}, {'name': 'cc-number'}, {'name': 'creditcard'},
    {'name': 'cvv'}, {'name': 'cvc'}, {'id': 'security-code'},
    {'name': 'ssn'}, {'label': 'Social Security Number'},
    {'name': 'otp'}, {'name': 'totp'}, {'label': 'One-time code'},
    {'name': 'api_key'}, {'name': 'private_key'}, {'label': 'Seed phrase'},
    {'name': 'passport'}, {'name': 'pin'},
]


@pytest.mark.parametrize('field', SECRET_FIELDS)
def test_secret_fields_are_refused(field):
    risk, _ = classify_field(el(tag='input', **field))
    assert risk is Risk.CREDENTIAL, field


ORDINARY_FIELDS = [
    {'name': 'email'}, {'name': 'search'}, {'name': 'quantity'},
    {'name': 'address_line_1'}, {'name': 'first_name'}, {'label': 'Delivery notes'},
    {'name': 'coupon'}, {'name': 'postcode'},
]


@pytest.mark.parametrize('field', ORDINARY_FIELDS)
def test_ordinary_fields_are_typeable(field):
    assert classify_field(el(tag='input', **field))[0] is not Risk.CREDENTIAL, field


def test_a_password_field_is_caught_by_type_even_with_an_innocent_name():
    assert classify_field(el(tag='input', type='password', name='q'))[0] is Risk.CREDENTIAL


FREE_TRIALS = [
    'Start my 30 day free trial',
    'Start my free trial',
    'Start your free 14-day trial',
    'Start free trial',
    'Try free for 30 days',
    'Begin your premium membership',
]


@pytest.mark.parametrize('label', FREE_TRIALS)
def test_free_trials_are_purchases(label):
    """The most common way an autonomous agent commits someone to a recurring
    charge, because the button never says "pay". Missed on a real page with a
    tighter pattern than this."""
    risk, _ = classify_click(el(label=label))
    assert risk is Risk.PURCHASE, label


ARTICLED = [
    'Place an order', 'Place the order', 'Submit an order', 'Submit the order',
    'Complete purchase', 'Charge the card on file', 'Charge the customer',
]


@pytest.mark.parametrize('label', ARTICLED)
def test_articles_do_not_hide_a_purchase(label):
    """Found by a real MCP tool description: `place an order` did not match a
    pattern written as `place (your)? order`. The article between verb and noun
    is the commonest phrasing there is."""
    assert classify_click(el(label=label))[0] is Risk.PURCHASE, label


NOT_BUYING_BUT_MENTIONS_ORDERS = [
    'Search orders', 'View order history', 'Order details', 'Sort by order date',
]


@pytest.mark.parametrize('label', NOT_BUYING_BUT_MENTIONS_ORDERS)
def test_merely_mentioning_orders_is_not_buying(label):
    """The widened pattern must not swallow every screen with the word order
    on it, or the prompt fires constantly and stops being read."""
    assert classify_click(el(label=label))[0] is not Risk.PURCHASE, label

"""Outbound plumbing: how a provider's traffic leaves this machine."""

from openmirror.net.transport import DIRECT, Transport, transport_for

__all__ = ['Transport', 'DIRECT', 'transport_for']

# Order Status and Changes

This page explains what each order status means and how long an order stays
changeable.

## The five statuses

| Status | What it means |
| --- | --- |
| Received | The order is in the system. Payment has not been taken. |
| Confirmed | Payment is authorised and the order is queued for picking. |
| Picking | The warehouse has started. The order can no longer be changed. |
| Dispatched | The parcel has left the warehouse and has a tracking number. |
| Closed | Delivered, or returned and refunded. |

An order moves from Received to Confirmed within minutes on working days. If it
stays on Received for more than two hours, the payment authorisation did not
come back and the order needs a new payment method.

## The change window

An order can be changed for **60 minutes** after it reaches Confirmed, or until
it reaches Picking, whichever comes first. Orders confirmed after 13:00 keep the
full 60 minutes, because picking does not start until the next working day.

Within the window you can change:

- the delivery address, as long as the zone does not change,
- the service, from Standard up to Tracked,
- quantities, downwards only,
- and you can remove a line.

Changing the zone is not possible, because the shipping cost and the transit
time were calculated for the original zone. Cancel and order again.

Adding a line is not possible either. A second order is the faster route and
ships together when both reach Picking on the same day.

## Cancelling

An order can be cancelled at any point before Dispatched, including during
Picking. A cancellation during Picking is a request rather than an instruction:
it succeeds if the picker has not yet reached the packing bench.

Once the status is Dispatched, cancelling is no longer possible. Refuse the
parcel at the door or register a return.

A cancelled order releases the payment authorisation the same working day. The
release can still take a few days to appear, depending on the card issuer.

## Tracking

The dispatch confirmation carries the tracking number. Tracking becomes live
once the carrier scans the parcel into its network, usually the evening of
dispatch.

A tracking number that shows nothing the next morning normally means the parcel
missed the evening collection and travels a day later. The transit time then
starts a day later as well.

## Split deliveries

An order that mixes oversized freight with normal items travels together and
arrives together. An order where one line is on back order does not: the
available lines dispatch first, and the back-ordered line follows without a
second shipping charge.

The order shows as Dispatched once the first parcel leaves. Each parcel carries
its own tracking number.

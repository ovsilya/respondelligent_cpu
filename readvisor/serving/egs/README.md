Examples in this directory have been used for testing the
deployed model.

# Greetings and Salutations

The current database structure does not allow to extract
fully customized greetings and salutations simply. At
present we provide **TWO** options:

1. Given the current values from the columns `greetingtype`
   and `goodbyetext` from customer information table
   `SF-guard-group` (see `*210624.json` files), infer an appropriate
   greeting/salutation based on a predefined list of
   allowable values (see `parse_greetingtype()` and `parsegoodbytext()` in `IO_utils.py`).

2. Given a list of allowable greetings and salutations for a
   particular customer (see `*210512.json` files), randomly
   select 1 of each. Ideally these should come from the
   customer information table in the database.

We recommend the latter as a more reliable, structured and future-proof solution.
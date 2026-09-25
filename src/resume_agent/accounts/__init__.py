"""Accounts, per-user storage and sign-in: the multi-user mode (M10).

Imported in every mode, because `api/app.py` has to be able to choose it, but
**inert unless multi-user mode is switched on**. None of its three dependencies
load in local mode: authlib and Starlette's session middleware are imported
inside the functions that install them, and psycopg only when a Postgres URL is
opened. Running `resume-agent` on your own machine behaves exactly as it did
before this package existed.
"""

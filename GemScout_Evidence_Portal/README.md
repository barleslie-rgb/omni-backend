# Gem Scout Evidence Review Portal — V2

This is a standalone static portal. It does not edit `main.py` or the Flutter screens.

## Configure
Open `index.html` and replace only `REPLACE_WITH_SUPABASE_PROJECT_URL` and `REPLACE_WITH_SUPABASE_PUBLIC_ANON_KEY` with your existing Supabase Project URL and public anon/publishable key. Never use the Supabase service-role/secret key in browser code.

## Form behavior
- Selecting a candidate is required.
- Business name, phone, website, photo URLs, source URLs, and notes are individually optional.
- A submission must contain at least one useful detail (corrected name, phone, website, photo URL, or source URL).
- Review approval still requires at least one source URL and at least one phone, website, or photo URL.
- It does not publish candidates to Community Gems.

Deploy as a separate static site. The existing backend and Flutter screens need not be modified.

> Working plan for moving the NVMAI wiki to the Discourse forum at nvmai.discourse.group.
> Status: draft — decide the forum structure and login flow, then publish article by article.

## Forum state (checked 2026-09-10)

- Live, stock Discourse, title "NVMAI". Owner account: `Pummelchen`.
- Categories: `General` (blue book), `Site Feedback`, `Uncategorized`.
- Topics: only the two auto-generated ones (`Welcome to NVMAI!`, `About the General category`).
- Public JSON API works unauthenticated (`/site.json`, `/latest.json`, topic JSON).
- Posting requires admin credentials: an API Key (Admin panel → Users → Pummelchen → Credentials), used with `Authorization: ApiKey <key>`.
- Login is Google sign-in (site config, already set up by the owner).

## Structure proposal

Create these categories (order in the sidebar):

| Category | Icon | Contents |
| --- | --- | --- |
| **Guides** | 📗 | The feature articles of this series |
| **Reference** | 📕 | System design, API reference, release notes |
| **General** | 📘 | Keep for discussion |
| **Show & Tell** | 🖼️ | User benchmarks, setups, results |
| **Site Feedback** | 💬 | Already exists |

Each wiki page becomes one **topic** in Guides or Reference, with a pinned "Series" topic at the top of Guides that links everything, in reading order.

## Article series (wiki source → forum article)

Status: ⬜ not started · ✍️ drafted · ✅ published

| # | Title | Category | Wiki source | Status |
| --- | --- | --- | --- | --- |
| 01 | What NVMAI is, and what it is made for | Guides | Home + Features | ✍️ drafted (`docs/site/01-what-is-nvmai.md`) |
| 02 | Getting Started: install and first run | Guides | Getting-Started | ⬜ |
| 03 | SSD expert streaming: how it works | Guides | System-Design (expert streaming) + v4.1/v4.2 docs | ⬜ |
| 04 | The RAM budget and bounded expert cache | Guides | System-Design (memory) + Runtime-Controls | ⬜ |
| 05 | Installs and verified receipts | Guides | Getting-Started (receipts) + AGENTS.md rules | ⬜ |
| 06 | The OpenAI-compatible server | Guides | OpenAI-Compatible-Server | ⬜ |
| 07 | Runtime controls | Guides | Runtime-Controls | ⬜ |
| 08 | Long context: RoPE and YaRN, KV compression | Guides | Runtime-Controls (context) + v4.5 docs | ⬜ |
| 09 | ANE prefill and the Metal engine | Guides | v4.5-ane-prefill + v4.6-optimization-inventory | ⬜ |
| 10 | Agent memory | Guides | agent-memory.md | ⬜ |
| 11 | Benchmarking: how to measure | Reference | Benchmarking-Guide + Benchmarks | ⬜ |
| 12 | System design (overview) | Reference | System-Design + v4-core-design | ⬜ |
| 13 | FAQ | General | FAQ | ⬜ |
| 14 | Release notes | Reference | Changelog + release-notes-v5.x | ⬜ |

## Workflow

1. Author each article as `docs/site/NN-slug.md` in the repo (canonical, reviewable, diffs against the wiki).
2. Review the draft here; fix inaccuracies against the repo before posting.
3. Publish to the forum via the Discourse API with the owner's API key:
   `POST /topics.json` with `title`, `category_id`, `raw` (or a script that reads the markdown file).
   Discourse converts markdown natively; code blocks and tables carry over.
4. Mark the row in this table ✅ with the topic URL.
5. Commit `docs/site/` + this plan to git after each published article.

## Style rules for the articles

- Short paragraphs, plain words. A forum reader is skimming, not studying.
- Every number gets its source: the benchmark script, the release note, or the machine it ran on.
- One article = one feature. Cross-link instead of duplicating.
- Limits get the same space as features. A limit that surprises later is a support ticket.
- No copy/paste from the wiki: the wiki is the spec, the article is the explanation.
- End every article with a "where to go next" link into the series.

## Open questions for the owner

1. Categories: keep the proposal, or start simpler (Guides + Reference only)?
2. Should the wiki be marked "moved to the forum" with a banner, or kept in sync?
3. Google login: is the site already using the Google OAuth plugin, or does the owner need to install/configure it in the admin panel?
4. API key: generate one and hand it over (it can be revoked in the admin panel at any time).
5. Pinned series topic: title it "NVMAI documentation" and pin it to the Guides category?

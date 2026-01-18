# Discord Bot Integration Notes

This document tracks integration details and coordination between the Discord bot and the Recommendation Engine API.

---

## Status: Active Integration (January 2026)

### Endpoints Being Used

| Endpoint | Bot Usage | Status |
|----------|-----------|--------|
| `GET /api/v1/recommendations` | Main flip command | ✅ Active |
| `GET /api/v1/recommendations/item/{id}` | Item lookup | ✅ Active |
| `GET /api/v1/items/search` | Autocomplete for `/report`, `/item` | ✅ Active |
| `POST /api/v1/recommendations/{rec_id}/outcome` | Trade outcome reporting | ✅ Active |

---

## Implemented Features

### 1. Item Search Endpoint

**Endpoint:** `GET /api/v1/items/search?q={query}&limit=10`

**Bot Usage:**
- Autocomplete in `/report` command
- `/item` command for looking up specific items

**Features:**
- [x] Fuzzy matching handles partial names (e.g., "armadyl" → "Armadyl godsword")
- [x] Case-insensitive search
- [x] Typo tolerance via rapidfuzz library

### 2. Trade Outcome Reporting

**Endpoint:** `POST /api/v1/recommendations/{rec_id}/outcome`

**Bot Behavior:**
- Sends trade outcomes when users mark trades as "Filled"
- User IDs are **SHA256 hashed** before sending (privacy)

**Privacy Guarantees:**
- No Discord IDs or PII sent
- Data used only for model training

**Note:** Requires `OUTCOME_DB_CONNECTION_STRING` environment variable to be configured. If not set, outcome recording is disabled (returns 503).

### 3. Crowding & Exclusion Parameters

**Parameters now being passed:**

```
GET /api/v1/recommendations?capital=10000000&user_id=sha256_hash&exclude=rec_123&exclude_item_ids=536,5295
```

| Parameter | Purpose |
|-----------|---------|
| `user_id` | Crowding tracking (limits concurrent users on same items) |
| `exclude` | Filter previously-seen recommendation IDs |
| `exclude_item_ids` | Filter items user is already trading |

**Status:**
- [x] `exclude_item_ids` implemented (PR #33)
- [x] Crowding tracker active

---

## Open Question: OSRS Acronym Handling

### The Ask

Does the item search fuzzy matching handle common OSRS acronyms?

| Acronym | Expected Match |
|---------|----------------|
| `ags` | Armadyl godsword |
| `bgs` | Bandos godsword |
| `sgs` | Saradomin godsword |
| `zgs` | Zamorak godsword |
| `dhl` | Dragon hunter lance |
| `dhcb` | Dragon hunter crossbow |
| `bp` | Toxic blowpipe |
| `tbow` | Twisted bow |
| `scythe` | Scythe of vitur |
| `sang` | Sanguinesti staff |
| `acb` | Armadyl crossbow |
| `dcb` | Dragon crossbow |
| `dfs` | Dragonfire shield |
| `bcp` | Bandos chestplate |
| `tassets` | Bandos tassets |
| `prims` | Primordial boots |
| `pegs` | Pegasian boots |
| `eternals` | Eternal boots |
| `zenny` | Zenyte shard |
| `dex` | Dexterous prayer scroll |
| `arcane` | Arcane prayer scroll |
| `ely` | Elysian spirit shield |
| `arma` | Armadyl (any piece) |
| `bandos` | Bandos (any piece) |
| `torva` | Torva (any piece) |
| `ancestral` | Ancestral (any piece) |
| `inq` | Inquisitor's (any piece) |
| `faceguard` | Neitiznot faceguard |
| `rapier` | Ghrazi rapier |
| `kodai` | Kodai wand |

### Current Behavior

**Testing Required** - The search endpoint needs to be verified with acronym queries.

### Recommendation

If acronym expansion is not currently supported, two options:

1. **API-side (preferred):** Add acronym lookup table in `src/api.py` search endpoint
2. **Bot-side fallback:** Discord bot expands acronyms before calling API

---

## Contact

- **Bot Team:** [Discord bot repository issues]
- **Engine Team:** [This repository issues](https://github.com/MarcusFranz/gept-recommendation-engine/issues)

import type { APIRoute } from 'astro';
import { searchItems } from '../../../lib/api';

// Common OSRS item acronyms
const acronyms: Record<string, string> = {
  'ags': 'Armadyl Godsword',
  'bgs': 'Bandos Godsword',
  'sgs': 'Saradomin Godsword',
  'zgs': 'Zamorak Godsword',
  'dbow': 'Dark Bow',
  'dclaws': 'Dragon Claws',
  'dwh': 'Dragon Warhammer',
  'tbow': 'Twisted Bow',
  'scythe': 'Scythe of Vitur',
  'ely': 'Elysian Spirit Shield',
  'arcane': 'Arcane Spirit Shield',
  'spectral': 'Spectral Spirit Shield',
  'dfs': 'Dragonfire Shield',
  'bcp': 'Bandos Chestplate',
  'tassets': 'Bandos Tassets',
  'prims': 'Primordial Boots',
  'pegs': 'Pegasian Boots',
  'eternals': 'Eternal Boots',
  'tort': 'Tormented Bracelet',
  'anguish': 'Necklace of Anguish',
  'torture': 'Amulet of Torture',
  'fury': 'Amulet of Fury',
  'seren': 'Blade of Saeldor',
  'sang': 'Sanguinesti Staff',
  'rapier': 'Ghrazi Rapier',
  'inquis': 'Inquisitor\'s Mace'
};

// Mock item database for demo
const mockItems = [
  { id: 11802, name: 'Armadyl Godsword', acronym: 'AGS' },
  { id: 11804, name: 'Bandos Godsword', acronym: 'BGS' },
  { id: 11806, name: 'Saradomin Godsword', acronym: 'SGS' },
  { id: 11808, name: 'Zamorak Godsword', acronym: 'ZGS' },
  { id: 536, name: 'Dragon Bones' },
  { id: 20997, name: 'Twisted Bow', acronym: 'TBow' },
  { id: 22325, name: 'Scythe of Vitur', acronym: 'Scythe' },
  { id: 12825, name: 'Elysian Spirit Shield', acronym: 'Ely' },
  { id: 12827, name: 'Arcane Spirit Shield', acronym: 'Arcane' },
  { id: 11283, name: 'Dragonfire Shield', acronym: 'DFS' },
  { id: 11832, name: 'Bandos Chestplate', acronym: 'BCP' },
  { id: 11834, name: 'Bandos Tassets', acronym: 'Tassets' },
  { id: 13239, name: 'Primordial Boots', acronym: 'Prims' },
  { id: 13237, name: 'Pegasian Boots', acronym: 'Pegs' },
  { id: 13235, name: 'Eternal Boots', acronym: 'Eternals' },
  { id: 19544, name: 'Tormented Bracelet', acronym: 'Tort' },
  { id: 19547, name: 'Necklace of Anguish', acronym: 'Anguish' },
  { id: 19553, name: 'Amulet of Torture', acronym: 'Torture' },
  { id: 6585, name: 'Amulet of Fury', acronym: 'Fury' },
  { id: 22324, name: 'Ghrazi Rapier', acronym: 'Rapier' },
  { id: 22323, name: 'Sanguinesti Staff', acronym: 'Sang' },
  { id: 13576, name: 'Dragon Warhammer', acronym: 'DWH' },
  { id: 13652, name: 'Dragon Claws', acronym: 'DClaws' },
  { id: 11235, name: 'Dark Bow', acronym: 'DBow' },
  { id: 560, name: 'Death Rune' },
  { id: 565, name: 'Blood Rune' },
  { id: 4151, name: 'Abyssal Whip', acronym: 'Whip' },
  { id: 12002, name: 'Occult Necklace', acronym: 'Occult' },
  { id: 21034, name: 'Ancestral Robe Top' },
  { id: 21036, name: 'Ancestral Robe Bottom' },
  { id: 21018, name: 'Ancestral Hat' }
];

export const GET: APIRoute = async ({ url }) => {
  const query = url.searchParams.get('q')?.toLowerCase() || '';
  const rawLimit = parseInt(url.searchParams.get('limit') || '15');
  const limit = Math.min(Math.max(1, rawLimit || 15), 50); // Cap between 1-50

  // Common cache headers for search results (5 min cache, 10 min stale-while-revalidate)
  const cacheHeaders = {
    'Content-Type': 'application/json',
    'Cache-Control': 'public, max-age=300, stale-while-revalidate=600'
  };

  if (query.length < 2) {
    return new Response(JSON.stringify({
      success: true,
      data: []
    }), {
      status: 200,
      headers: cacheHeaders
    });
  }

  try {
    // First try to get results from API
    const apiResults = await searchItems(query, limit);
    if (apiResults.length > 0) {
      return new Response(JSON.stringify({
        success: true,
        data: apiResults
      }), {
        status: 200,
        headers: cacheHeaders
      });
    }
  } catch {
    // Fall through to mock data
  }

  // Expand acronym if applicable
  const expandedQuery = acronyms[query] || query;

  // Search mock items
  const results = mockItems
    .filter(item => {
      const nameMatch = item.name.toLowerCase().includes(expandedQuery.toLowerCase());
      const acronymMatch = item.acronym?.toLowerCase() === query;
      return nameMatch || acronymMatch;
    })
    .slice(0, limit);

  return new Response(JSON.stringify({
    success: true,
    data: results
  }), {
    status: 200,
    headers: cacheHeaders
  });
};

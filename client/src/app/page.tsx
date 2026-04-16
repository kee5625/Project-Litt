"use client";

import { FormEvent, useMemo, useState } from "react";

type ResearchResult = {
  id: string;
  score: number;
  case_name?: string | null;
  citation?: string | null;
  court?: string | null;
  jurisdiction?: string | null;
  court_level?: string | null;
  year?: number | null;
  outcome?: string | null;
  is_good_law?: boolean | null;
  source_url?: string | null;
  holding_text?: string | null;
  full_cite_str?: string | null;
  match_reasons?: string[];
};

type SearchResponse = {
  query: string;
  applied_filters: Record<string, string | number | boolean>;
  count: number;
  latency_ms: number;
  components?: {
    vector_count?: number;
    keyword_count?: number;
    keyword_terms?: string[];
  };
  results: ResearchResult[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function scorePercent(score: number, maxScore: number): number {
  if (!Number.isFinite(score) || !Number.isFinite(maxScore) || maxScore <= 0) {
    return 0;
  }
  const normalized = Math.max(0, Math.min(1, score / maxScore));
  return Math.round(normalized * 100);
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [jurisdiction, setJurisdiction] = useState("all");
  const [courtLevel, setCourtLevel] = useState("all");
  const [yearMin, setYearMin] = useState("2010");
  const [goodLawOnly, setGoodLawOnly] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);

  const hasResults = (response?.results?.length ?? 0) > 0;
  const sortedResults = useMemo(
    () => [...(response?.results ?? [])],
    [response?.results],
  );
  const maxScore = useMemo(() => {
    return sortedResults.reduce((best, item) => {
      return Number.isFinite(item.score) ? Math.max(best, item.score) : best;
    }, 0);
  }, [sortedResults]);

  async function onSearch(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      params.set("query", trimmed);
      params.set("limit", "5");

      if (jurisdiction !== "all") params.set("jurisdiction", jurisdiction);
      if (courtLevel !== "all") params.set("court_level", courtLevel);
      if (yearMin.trim()) params.set("year_from", yearMin.trim());
      if (goodLawOnly) params.set("good_law_only", "true");

      const res = await fetch(`${API_BASE}/api/search?${params.toString()}`, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      });

      if (!res.ok) {
        throw new Error(`Search failed (${res.status})`);
      }

      const data = (await res.json()) as SearchResponse;
      setResponse(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
      setResponse(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="panel">
        <header className="panelHeader">
          <h1>AI Case Law Research</h1>
          <p>Natural language search across grounded legal authorities.</p>
        </header>

        <form className="searchForm" onSubmit={onSearch}>
          <label className="queryLabel" htmlFor="query">
            Research question
          </label>
          <textarea
            id="query"
            name="query"
            placeholder="What is the standard for summary judgment in employment cases, 9th Circuit?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
          />

          <div className="filtersRow">
            <label>
              Jurisdiction
              <select
                value={jurisdiction}
                onChange={(e) => setJurisdiction(e.target.value)}
              >
                <option value="all">All jurisdictions</option>
                <option value="federal">Federal</option>
                <option value="state">State</option>
                <option value="district">District</option>
              </select>
            </label>

            <label>
              Court level
              <select
                value={courtLevel}
                onChange={(e) => setCourtLevel(e.target.value)}
              >
                <option value="all">All court levels</option>
                <option value="supreme">Supreme</option>
                <option value="circuit">Circuit</option>
                <option value="district">District</option>
                <option value="state">State</option>
              </select>
            </label>

            <label>
              Year min
              <input
                type="number"
                min={1900}
                max={2100}
                value={yearMin}
                onChange={(e) => setYearMin(e.target.value)}
              />
            </label>

            <label className="goodLawToggle">
              <input
                type="checkbox"
                checked={goodLawOnly}
                onChange={(e) => setGoodLawOnly(e.target.checked)}
              />
              Good law only
            </label>
          </div>

          <div className="actionsRow">
            <button type="submit" disabled={isLoading || !query.trim()}>
              {isLoading ? "Searching..." : "Search"}
            </button>
            {response ? (
              <span className="meta">
                {response.count} results ({(response.latency_ms / 1000).toFixed(1)}s)
              </span>
            ) : null}
          </div>
        </form>

        {error ? <p className="error">{error}</p> : null}

        <section className="resultsSection" aria-live="polite">
          {!hasResults && !isLoading ? (
            <p className="emptyState">Run a search to view ranked authorities.</p>
          ) : null}

          {sortedResults.map((item) => {
            const pct = scorePercent(item.score, maxScore);
            const title = item.case_name || item.full_cite_str || "Untitled authority";
            const subtitle = [item.citation, item.court, item.year]
              .filter(Boolean)
              .join(" • ");

            return (
              <article key={item.id} className="resultCard">
                <div className="resultHeader">
                  <h2>{title}</h2>
                  <span className={`status ${item.is_good_law ? "good" : "warn"}`}>
                    {item.is_good_law ? "good law" : "check status"}
                  </span>
                </div>

                <p className="subline">{subtitle}</p>

                <div className="scoreRow">
                  <div className="scoreBar" role="img" aria-label={`Relevance ${pct}%`}>
                    <div className="scoreFill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="scoreText">{pct}% relevance</span>
                </div>

                <p className="snippet">{item.holding_text ?? "No summary snippet available."}</p>

                <div className="tagRow">
                  {(item.match_reasons ?? []).map((tag) => (
                    <span key={`${item.id}-${tag}`} className="tag">
                      matched: {tag}
                    </span>
                  ))}
                </div>

                <div className="cardActions">
                  <button type="button" className="draftButton" disabled>
                    + Add to Draft
                  </button>
                  {item.source_url ? (
                    <a href={item.source_url} target="_blank" rel="noreferrer">
                      Open Source
                    </a>
                  ) : null}
                </div>
              </article>
            );
          })}
        </section>
      </section>
    </main>
  );
}

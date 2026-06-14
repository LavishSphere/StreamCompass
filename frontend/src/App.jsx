/**
 * App.jsx — StreamCompass browse / results page
 *
 * Rendered at "/browse". Reads the ?q= URL param set by LandingPage
 * and displays mock recommendation results in horizontal scrollable rows
 * grouped by type and quality.
 *
 * ── MOCK DATA MODE ──────────────────────────────────────────────────────────
 * This file currently uses hardcoded MOCK_DATA instead of calling the backend.
 * This is intentional — it lets you develop and test the UI locally without
 * needing the backend running or dealing with CORS.
 *
 * When you're ready to connect the real API, search for "TODO: API INTEGRATION"
 * comments throughout this file. Each one marks exactly what to change.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * Future API endpoints (when integrating):
 *   POST /recommend  — main ML recommendation pipeline (8-stage, see recommender.py)
 *   GET  /title/{t}  — full metadata for a single title (description, cast, director)
 *
 * External dependencies:
 *   react-router-dom — useSearchParams (read ?q=), useNavigate (logo → home)
 */

import { useState, useRef, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

/**
 * TODO: API INTEGRATION
 * Uncomment this when connecting the real backend:
 *   const API_BASE = "https://api.khayrul.com";
 */

/**
 * Maps platform keys to display labels and brand colors.
 * Keys match the field names the backend returns (netflix, hulu, etc.).
 */
const PLATFORM_COLORS = {
  netflix:     { bg: "#E50914", label: "Netflix"  },
  hulu:        { bg: "#1CE783", label: "Hulu"     },
  prime_video: { bg: "#00A8E0", label: "Prime"    },
  disney_plus: { bg: "#113CCF", label: "Disney+"  },
};

/** Platform filter chips shown in the nav bar. "All" means no filter. */
const PLATFORM_FILTER_OPTIONS = ["All", "Netflix", "Hulu", "Prime Video", "Disney+"];

/**
 * Maps human-readable chip label to the platform key used in filtering.
 * TODO: API INTEGRATION — this maps to the `platforms` field in POST /recommend body.
 */
const PLATFORM_KEY_MAP = {
  "Netflix":     "netflix",
  "Hulu":        "hulu",
  "Prime Video": "prime_video",
  "Disney+":     "disney_plus",
};

// ---------------------------------------------------------------------------
// Mock data
// Mock data representing the shape of what POST /recommend returns.
// Each field maps directly to the real API response schema so swapping
// in real data later requires no changes to the components below.
//
// TODO: API INTEGRATION — delete MOCK_DATA and MOCK_DETAIL when connecting
// the real backend. The components below already handle the same data shape.
// ---------------------------------------------------------------------------

const MOCK_DATA = [
  {
    title: "Inception", year: 2010, content_type: "movie",
    genres: "Action, Science Fiction, Thriller",
    imdb_score: 8.8, similarity_score: 0.98,
    netflix: 1, hulu: 0, prime_video: 1, disney_plus: 0,
  },
  {
    title: "Interstellar", year: 2014, content_type: "movie",
    genres: "Adventure, Drama, Science Fiction",
    imdb_score: 8.6, similarity_score: 0.94,
    netflix: 0, hulu: 1, prime_video: 1, disney_plus: 0,
  },
  {
    title: "The Matrix", year: 1999, content_type: "movie",
    genres: "Action, Science Fiction",
    imdb_score: 8.7, similarity_score: 0.91,
    netflix: 1, hulu: 0, prime_video: 0, disney_plus: 0,
  },
  {
    title: "Arrival", year: 2016, content_type: "movie",
    genres: "Drama, Science Fiction",
    imdb_score: 7.9, similarity_score: 0.89,
    netflix: 0, hulu: 1, prime_video: 1, disney_plus: 0,
  },
  {
    title: "Severance", year: 2022, content_type: "tv",
    genres: "Drama, Mystery, Science Fiction",
    imdb_score: 8.7, similarity_score: 0.88,
    netflix: 0, hulu: 0, prime_video: 0, disney_plus: 0,
  },
  {
    title: "Blade Runner 2049", year: 2017, content_type: "movie",
    genres: "Drama, Science Fiction",
    imdb_score: 8.0, similarity_score: 0.87,
    netflix: 0, hulu: 1, prime_video: 1, disney_plus: 0,
  },
  {
    title: "Ex Machina", year: 2014, content_type: "movie",
    genres: "Drama, Science Fiction, Thriller",
    imdb_score: 7.7, similarity_score: 0.85,
    netflix: 0, hulu: 0, prime_video: 1, disney_plus: 0,
  },
  {
    title: "Westworld", year: 2016, content_type: "tv",
    genres: "Drama, Science Fiction, Western",
    imdb_score: 8.5, similarity_score: 0.83,
    netflix: 0, hulu: 0, prime_video: 0, disney_plus: 0,
  },
  {
    title: "Annihilation", year: 2018, content_type: "movie",
    genres: "Horror, Science Fiction",
    imdb_score: 6.8, similarity_score: 0.82,
    netflix: 1, hulu: 0, prime_video: 0, disney_plus: 0,
  },
  {
    title: "Dark", year: 2017, content_type: "tv",
    genres: "Crime, Mystery, Science Fiction, Thriller",
    imdb_score: 8.8, similarity_score: 0.81,
    netflix: 1, hulu: 0, prime_video: 0, disney_plus: 0,
  },
  {
    title: "Stranger Things", year: 2016, content_type: "tv",
    genres: "Drama, Fantasy, Horror, Mystery",
    imdb_score: 8.7, similarity_score: 0.78,
    netflix: 1, hulu: 0, prime_video: 0, disney_plus: 0,
  },
  {
    title: "The Midnight Club", year: 2022, content_type: "tv",
    genres: "Drama, Horror, Mystery",
    imdb_score: 7.1, similarity_score: 0.70,
    netflix: 1, hulu: 0, prime_video: 0, disney_plus: 0,
  },
];

/**
 * Mock detail data for the drawer — simulates GET /title/{name} responses.
 * Keyed by lowercase title. Each entry adds description, cast, and director
 * on top of the base data already in MOCK_DATA.
 *
 * TODO: API INTEGRATION — delete this and replace with a real fetch in handleSelect.
 */
const MOCK_DETAIL = {
  "inception": {
    description: "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
    cast: "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
    director: "Christopher Nolan",
  },
  "interstellar": {
    description: "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
    cast: "Matthew McConaughey, Anne Hathaway, Jessica Chastain, Michael Caine",
    director: "Christopher Nolan",
  },
  "the matrix": {
    description: "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
    cast: "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving",
    director: "Lana Wachowski, Lilly Wachowski",
  },
  "arrival": {
    description: "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecrafts appear around the world.",
    cast: "Amy Adams, Jeremy Renner, Forest Whitaker",
    director: "Denis Villeneuve",
  },
  "severance": {
    description: "Mark leads a team of office workers whose memories have been surgically divided between their work and personal lives.",
    cast: "Adam Scott, Britt Lower, Zach Cherry, Patricia Arquette",
    director: "Ben Stiller",
  },
  "blade runner 2049": {
    description: "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard.",
    cast: "Ryan Gosling, Harrison Ford, Ana de Armas, Sylvia Hoeks",
    director: "Denis Villeneuve",
  },
  "ex machina": {
    description: "A programmer is selected to participate in a ground-breaking experiment in artificial intelligence by evaluating the human qualities of a robot.",
    cast: "Alicia Vikander, Domhnall Gleeson, Oscar Isaac",
    director: "Alex Garland",
  },
  "dark": {
    description: "A family saga with a supernatural twist, set in a German town where the disappearance of two young children exposes the double lives of four families.",
    cast: "Louis Hofmann, Oliver Masucci, Karoline Eichhorn",
    director: "Baran bo Odar",
  },
  "stranger things": {
    description: "When a young boy disappears, his mother, a police chief and his friends must confront terrifying supernatural forces in order to get him back.",
    cast: "Millie Bobby Brown, Winona Ryder, David Harbour, Finn Wolfhard",
    director: "The Duffer Brothers",
  },
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Extracts which streaming platforms a result item is available on.
 * The API (and mock data) use binary flags (0 or 1) per platform column.
 *
 * @param {Object} item - A result object from MOCK_DATA or the real API
 * @returns {string[]} Array of platform keys, e.g. ["netflix", "hulu"]
 */
function getPlatforms(item) {
  const plats = [];
  if (item.netflix === 1)     plats.push("netflix");
  if (item.hulu === 1)        plats.push("hulu");
  if (item.prime_video === 1) plats.push("prime_video");
  if (item.disney_plus === 1) plats.push("disney_plus");
  return plats;
}

/**
 * Filters mock results by active platform chip and search query.
 * Simulates what the backend does server-side when the real API is connected.
 *
 * TODO: API INTEGRATION — delete this function. Filtering moves to the
 * POST /recommend request body (platforms field + query string).
 *
 * @param {string} query        - Search string to filter titles by
 * @param {string} activeFilter - Active platform chip label (e.g. "Netflix")
 * @returns {Array} Filtered and sorted subset of MOCK_DATA
 */
function getFilteredMockResults(query, activeFilter) {
  let results = [...MOCK_DATA];

  // Filter by search query (case-insensitive substring match on title/genres)
  if (query.trim()) {
    const q = query.toLowerCase();
    results = results.filter(
      (r) =>
        r.title.toLowerCase().includes(q) ||
        r.genres.toLowerCase().includes(q)
    );

    // If no direct matches, return all mock data (simulates "related" results)
    if (results.length === 0) results = [...MOCK_DATA];
  }

  // Filter by platform chip
  const platformKey = PLATFORM_KEY_MAP[activeFilter];
  if (platformKey) {
    results = results.filter((r) => r[platformKey] === 1);
    if (results.length === 0) results = [...MOCK_DATA]; // fallback if nothing matches
  }

  return results;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Inline compass SVG used in the nav bar at 26px. */
const COMPASS_SVG = (
  <svg width="26" height="26" viewBox="0 0 727 727" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M726.286 382.265C716.048 579.042 550.857 731.808 353.88 726.652C156.909 721.499 -0.0698985 560.3 2.33495e-05 363.261C0.0699452 166.22 157.163 5.13451 354.138 0.119903C551.118 -4.89471 716.204 147.989 726.298 344.771C726.939 357.261 726.935 369.776 726.286 382.265ZM363.04 648.241C437.939 647.726 509.575 617.523 562.228 564.257C626.165 500.534 660.884 408.214 643.736 318.289C632.208 257.853 606.505 206.438 562.747 162.732C512.444 112.378 443.751 79.9413 371.876 80.0242C368.845 80.0203 365.81 80.0499 362.775 80.1128C346.039 80.7902 329.937 81.1175 313.342 83.8046C177.884 105.735 73.5098 241.516 78.3039 377.081C79.3363 406.276 85.8055 434.276 96.9274 461.101C113.551 504.499 136.274 538.394 170.763 569.976C223.302 618.429 291.601 646.23 363.04 648.241Z" fill="#00E5FF"/>
    <path d="M534.263 212.763L534.611 213.28C526.431 230.554 517.181 248.382 508.595 265.552L460.814 361.089C456.775 369.171 428.458 427.593 425.599 430.241C422.38 433.226 386.044 450.441 379.392 453.765L228.623 529.163C444.233 532.351 215.075 535.679 208.859 539.058L208.58 538.527L303.188 349.061C307.608 340.207 312.584 329.628 317.31 321.093L534.263 212.763ZM250.367 497.003C265.782 488.796 282.191 480.956 297.845 473.042L357.111 443.417C367.431 438.249 378.552 432.417 388.962 427.585C383.759 421.913 377.482 415.894 372.001 410.405L342.477 380.862C339.178 377.562 322.682 360.253 319.631 358.554C314.848 368.675 309.354 379.226 304.318 389.273L278.589 440.663C269.598 458.542 259.788 479.71 250.367 497.003Z" fill="#00E5FF"/>
  </svg>
);

/**
 * PlatformBadge — colored pill showing which streaming service a title is on.
 * @param {string} platform - Platform key (e.g. "netflix", "hulu")
 */
function PlatformBadge({ platform }) {
  const p = PLATFORM_COLORS[platform];
  if (!p) return null;
  return (
    <span style={{
      background: p.bg, color: "#fff", fontSize: "10px", fontWeight: 600,
      letterSpacing: "0.04em", padding: "3px 7px", borderRadius: "4px",
      whiteSpace: "nowrap", fontFamily: "inherit",
    }}>
      {p.label}
    </span>
  );
}

/**
 * StarRating — gold star icon + numeric IMDb score.
 * Returns null if no rating available.
 * @param {number|null} rating - IMDb score (0–10)
 */
function StarRating({ rating }) {
  if (!rating) return null;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
      <svg width="12" height="12" viewBox="0 0 12 12" fill="#F5A623">
        <polygon points="6,1 7.5,4.5 11,5 8.5,7.5 9.2,11 6,9.2 2.8,11 3.5,7.5 1,5 4.5,4.5" />
      </svg>
      <span style={{ fontSize: "12px", color: "#aaa", fontFamily: "inherit" }}>{rating}</span>
    </span>
  );
}

/**
 * TitleCard — individual movie/show card in a ScrollRow.
 *
 * Displays a monogram placeholder where a poster image would go.
 * When real poster URLs are available, swap the initials div for:
 *   <img src={item.poster_url} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
 *
 * Hover: lifts + scales, cyan top border, initials glow.
 * Active (drawer open): cyan outline ring.
 * Similarity badge shown only for scores >= 80%.
 *
 * @param {Object}   item     - Result object from MOCK_DATA or real API
 * @param {Function} onClick  - Called with item when card is clicked
 * @param {boolean}  isActive - Whether this card's drawer is currently open
 */
function TitleCard({ item, onClick, isActive }) {
  const [hovered, setHovered] = useState(false);
  const initials = item.title.split(" ").slice(0, 2).map((w) => w[0]).join("").toUpperCase();
  const platforms = getPlatforms(item);
  const simPct = item.similarity_score ? Math.round(item.similarity_score * 100) : null;

  return (
    <div
      onClick={() => onClick(item)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        flexShrink: 0, width: "148px", cursor: "pointer",
        transition: "transform 0.18s ease",
        transform: hovered ? "translateY(-4px) scale(1.03)" : "translateY(0) scale(1)",
        outline: isActive ? "2px solid #00E5FF" : "none",
        outlineOffset: "2px", borderRadius: "6px",
      }}
    >
      {/* Card face — 148×210px (roughly 2:3 poster ratio) */}
      <div style={{
        width: "148px", height: "210px",
        background: hovered ? "#1a1a1a" : "#111",
        border: hovered ? "1px solid #444" : "1px solid #1e1e1e",
        borderRadius: "6px", display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        position: "relative", overflow: "hidden",
        transition: "border-color 0.18s, background 0.18s",
      }}>
        {hovered && (
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: "#00E5FF", opacity: 0.8 }} />
        )}

        {/* Monogram placeholder — replace with <img> when poster URLs are available */}
        <div style={{
          width: "52px", height: "52px", borderRadius: "50%",
          background: "#1e1e1e", border: "1px solid #444",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "18px", fontWeight: 600,
          color: hovered ? "#00E5FF" : "#999",
          letterSpacing: "0.05em", transition: "color 0.18s",
          fontFamily: "inherit", marginBottom: "12px",
        }}>
          {initials}
        </div>

        {/* Title + year overlay at the bottom */}
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          padding: "12px 10px",
          background: "linear-gradient(to top, rgba(0,0,0,0.95) 0%, transparent 100%)",
        }}>
          <div style={{
            fontSize: "11px", fontWeight: 600, color: "#fff", lineHeight: 1.3,
            marginBottom: "5px", fontFamily: "inherit",
            overflow: "hidden", display: "-webkit-box",
            WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
          }}>
            {item.title}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "10px", color: "#999", fontFamily: "inherit" }}>{item.year}</span>
            <StarRating rating={item.imdb_score} />
          </div>
        </div>

        {/* Similarity badge — only shown for high-confidence matches */}
        {simPct >= 80 && (
          <div style={{
            position: "absolute", top: "8px", right: "8px",
            background: "rgba(0,229,255,0.15)", border: "1px solid rgba(0,229,255,0.4)",
            borderRadius: "3px", padding: "2px 5px",
            fontSize: "9px", fontWeight: 700, color: "#00E5FF",
            fontFamily: "inherit", letterSpacing: "0.05em",
          }}>
            {simPct}%
          </div>
        )}
      </div>

      {/* Genre label below the card — always shows genre, never platform names */}
      <div style={{
        marginTop: "8px", fontSize: "11px",
        color: hovered ? "#ccc" : "#999",
        fontFamily: "inherit", transition: "color 0.18s",
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", paddingLeft: "2px",
      }}>
        {(item.genres || "").split(",").slice(0, 2).map((g) => g.trim()).join(" · ")}
      </div>
    </div>
  );
}

/**
 * ScrollRow — labeled horizontal strip of TitleCards with arrow buttons.
 *
 * @param {string}   label    - Row heading
 * @param {Array}    items    - Cards to render
 * @param {Function} onSelect - Passed to each TitleCard onClick
 * @param {string}   activeId - Title of the currently open drawer card
 */
function ScrollRow({ label, items, onSelect, activeId }) {
  const ref = useRef(null);
  const scroll = (dir) => ref.current?.scrollBy({ left: dir * 600, behavior: "smooth" });

  return (
    <div style={{ marginBottom: "40px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
        <h2 style={{ fontSize: "15px", fontWeight: 600, color: "#fff", margin: 0, fontFamily: "inherit" }}>
          {label}
        </h2>
        <div style={{ display: "flex", gap: "6px" }}>
          {["‹", "›"].map((arrow, i) => (
            <button key={arrow} onClick={() => scroll(i === 0 ? -1 : 1)}
              style={{
                background: "transparent", border: "1px solid #2a2a2a", borderRadius: "4px",
                color: "#444", width: "28px", height: "28px", cursor: "pointer",
                fontSize: "16px", display: "flex", alignItems: "center", justifyContent: "center",
                transition: "border-color 0.15s, color 0.15s", fontFamily: "inherit", lineHeight: 1,
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#999"; e.currentTarget.style.color = "#fff"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#2a2a2a"; e.currentTarget.style.color = "#444"; }}
            >{arrow}</button>
          ))}
        </div>
      </div>
      {items.length === 0 ? (
        <p style={{ fontSize: "13px", color: "#999", fontFamily: "inherit" }}>No results found.</p>
      ) : (
        <div ref={ref} style={{ display: "flex", gap: "12px", overflowX: "auto", paddingBottom: "8px", scrollbarWidth: "none" }}>
          {items.map((item, idx) => (
            <TitleCard key={`${item.title}-${idx}`} item={item} onClick={onSelect} isActive={activeId === item.title} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * DetailDrawer — slide-up panel showing full metadata for a selected title.
 *
 * Returns null when no item is selected (zero DOM footprint when closed).
 * Mock mode: description/cast comes from MOCK_DETAIL lookup by title.
 * Real API mode: these fields come directly from GET /title/{name}.
 *
 * @param {Object|null} item    - Title detail object, or null to hide
 * @param {Function}    onClose - Called when the × button is clicked
 */
function DetailDrawer({ item, onClose }) {
  if (!item) return null;

  const platforms = getPlatforms(item);
  const genres = item.genres ? item.genres.split(",").map((g) => g.trim()).filter(Boolean) : [];

  return (
    <div style={{
      position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 100,
      background: "#0d0d0d", borderTop: "1px solid #444",
      padding: "28px 32px 32px",
      boxShadow: "0 -20px 60px rgba(0,0,0,0.8)",
      maxHeight: "50vh", overflowY: "auto",
      animation: "slideUp 0.22s cubic-bezier(0.22, 1, 0.36, 1)",
    }}>
      <style>{`
        @keyframes slideUp { from { transform: translateY(100%); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
      `}</style>
      <div style={{
        maxWidth: "900px", margin: "0 auto",
        display: "grid", gridTemplateColumns: "1fr auto",
        gap: "24px", alignItems: "start",
      }}>
        <div>
          <div style={{ marginBottom: "14px" }}>
            <h3 style={{ fontSize: "22px", fontWeight: 700, color: "#fff", margin: 0, letterSpacing: "-0.01em", fontFamily: "inherit" }}>
              {item.title}
            </h3>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "6px", flexWrap: "wrap" }}>
              {item.year && <span style={{ fontSize: "13px", color: "#444", fontFamily: "inherit" }}>{item.year}</span>}
              <span style={{ fontSize: "13px", color: "#444" }}>·</span>
              <span style={{ fontSize: "11px", border: "1px solid #444", borderRadius: "3px", padding: "1px 6px", color: "#888", fontFamily: "inherit" }}>
                {item.content_type === "tv" ? "TV Show" : "Movie"}
              </span>
              <StarRating rating={item.imdb_score} />
              {item.similarity_score && (
                <span style={{ fontSize: "11px", color: "#00E5FF", fontFamily: "inherit" }}>
                  {Math.round(item.similarity_score * 100)}% match
                </span>
              )}
            </div>
          </div>

          {item.description && (
            <p style={{ fontSize: "13px", color: "#aaa", lineHeight: 1.65, margin: "0 0 14px", fontFamily: "inherit" }}>
              {item.description}
            </p>
          )}

          {item.cast && (
            <div style={{ marginBottom: "12px" }}>
              <span style={{ fontSize: "11px", color: "#999", textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: "6px", fontFamily: "inherit", fontWeight: 600 }}>
                Cast
              </span>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {item.cast.split(",").slice(0, 5).map((name) => (
                  <span key={name} style={{ fontSize: "12px", color: "#ccc", background: "#161616", border: "1px solid #2a2a2a", borderRadius: "4px", padding: "3px 8px", fontFamily: "inherit" }}>
                    {name.trim()}
                  </span>
                ))}
              </div>
            </div>
          )}

          {genres.length > 0 && (
            <div style={{ marginBottom: "14px" }}>
              <span style={{ fontSize: "11px", color: "#999", textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: "6px", fontFamily: "inherit", fontWeight: 600 }}>
                Genres
              </span>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {genres.map((g) => (
                  <span key={g} style={{ fontSize: "11px", color: "#aaa", background: "#111", border: "1px solid #2a2a2a", borderRadius: "20px", padding: "3px 10px", fontFamily: "inherit" }}>
                    {g}
                  </span>
                ))}
              </div>
            </div>
          )}

          {platforms.length > 0 && (
            <div>
              <span style={{ fontSize: "11px", color: "#999", textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: "6px", fontFamily: "inherit", fontWeight: 600 }}>
                Available on
              </span>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {platforms.map((p) => <PlatformBadge key={p} platform={p} />)}
              </div>
            </div>
          )}
        </div>

        <button onClick={onClose}
          style={{
            background: "transparent", border: "1px solid #2a2a2a", borderRadius: "6px",
            color: "#444", width: "32px", height: "32px", cursor: "pointer",
            fontSize: "18px", display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0, transition: "border-color 0.15s, color 0.15s", fontFamily: "inherit",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#999"; e.currentTarget.style.color = "#fff"; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#2a2a2a"; e.currentTarget.style.color = "#444"; }}
        >×</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Browse page — main export
// ---------------------------------------------------------------------------

/**
 * App — the browse/results page, rendered at "/browse".
 *
 * State:
 *   query        {string}      - Active search term (from URL param or nav input)
 *   inputValue   {string}      - Live value in the nav search input
 *   selected     {Object|null} - Last-clicked card (with detail merged in)
 *   activeFilter {string}      - Active platform chip label
 *
 * TODO: API INTEGRATION — when connecting the real backend, replace the
 * getFilteredMockResults() call and MOCK_DETAIL lookup in handleSelect with:
 *   1. fetch(`${API_BASE}/recommend`, { method: "POST", body: ... })
 *   2. fetch(`${API_BASE}/title/${encodeURIComponent(item.title)}`)
 */
export default function App() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const initialQuery = searchParams.get("q") || "";

  const [query, setQuery]           = useState(initialQuery);
  const [inputValue, setInputValue] = useState(initialQuery);
  const [selected, setSelected]     = useState(null);
  /**
   * activePlatforms — Set of platform keys currently selected by the user.
   * Empty set means "All" (no platform filter). Multiple platforms can be
   * active simultaneously — clicking a chip toggles it on/off.
   *
   * Maps directly to the `platforms` array in POST /recommend request body.
   * TODO: API INTEGRATION — spread [...activePlatforms] into the request body.
   */
  const [activePlatforms, setActivePlatforms] = useState(new Set());

  /**
   * Toggles a platform chip on or off.
   * If the chip is already active, removes it. Otherwise adds it.
   *
   * @param {string} platformKey - e.g. "netflix", "hulu", "prime_video", "disney_plus"
   */
  const togglePlatform = (platformKey) => {
    setActivePlatforms((prev) => {
      const next = new Set(prev);
      if (next.has(platformKey)) {
        next.delete(platformKey);
      } else {
        next.add(platformKey);
      }
      return next;
    });
  };

  /**
   * Filter controls shown below the results headline.
   * In mock mode these filter the results array client-side.
   * TODO: API INTEGRATION — pass contentTypeFilter as `content_type` and
   * minImdbFilter as `min_imdb` in the POST /recommend request body instead.
   */
  const [contentTypeFilter, setContentTypeFilter] = useState("All");  // "All" | "Movies" | "TV Shows"
  const [minImdbFilter, setMinImdbFilter]         = useState("Any");  // "Any" | "6+" | "7+" | "8+" | "9+"

  /**
   * Derives results from mock data, then applies all active filters.
   * TODO: API INTEGRATION — replace with a useEffect that calls POST /recommend
   * and move all filtering to the request body.
   */
  let results = getFilteredMockResults(query, "All");

  // Apply platform filter — keep titles available on ANY of the selected platforms
  if (activePlatforms.size > 0) {
    results = results.filter((r) =>
      [...activePlatforms].some((p) => r[p] === 1)
    );
    if (results.length === 0) results = getFilteredMockResults(query, "All"); // fallback
  }

  if (contentTypeFilter === "Movies")   results = results.filter((r) => r.content_type === "movie");
  if (contentTypeFilter === "TV Shows") results = results.filter((r) => r.content_type === "tv");
  if (minImdbFilter !== "Any") {
    const minScore = parseFloat(minImdbFilter);
    results = results.filter((r) => r.imdb_score && r.imdb_score >= minScore);
  }

  /**
   * Handles card click — looks up mock detail and merges it onto the item.
   * TODO: API INTEGRATION — replace MOCK_DETAIL lookup with fetch to GET /title.
   */
  const handleSelect = (item) => {
    if (selected?.title === item.title) {
      setSelected(null);
      return;
    }
    const detail = MOCK_DETAIL[item.title.toLowerCase()] || {};
    setSelected({ ...item, ...detail });
  };

  /** Handles nav bar search form submission — updates query and URL param. */
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    setQuery(inputValue.trim());
    navigate(`/browse?q=${encodeURIComponent(inputValue.trim())}`, { replace: true });
  };

  // Derive row data by splitting filtered results into themed groups
  const topMatches   = results.slice(0, 15);
  const movieResults = results.filter((r) => r.content_type === "movie");
  const tvResults    = results.filter((r) => r.content_type === "tv");
  const highRated    = results.filter((r) => r.imdb_score && r.imdb_score >= 8.0);

  const rows = query ? [
    { label: `Best matches for "${query}"`,   items: topMatches   },
    ...(contentTypeFilter === "All" && movieResults.length > 0 ? [{ label: "Movies",             items: movieResults }] : []),
    ...(contentTypeFilter === "All" && tvResults.length    > 0 ? [{ label: "TV Shows",           items: tvResults    }] : []),
    ...(highRated.length    > 0 ? [{ label: "Highly rated picks", items: highRated    }] : []),
  ] : [];

  return (
    <div style={{
      minHeight: "100vh", background: "#000", color: "#fff",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif",
    }}>

      {/* ── Sticky navigation bar ── */}
      <header style={{
        position: "sticky", top: 0, zIndex: 50,
        background: "rgba(0,0,0,0.92)", backdropFilter: "blur(12px)",
        borderBottom: "1px solid #111", padding: "0 48px",
      }}>
        <div style={{
          maxWidth: "1600px", margin: "0 auto", height: "72px",
          display: "flex", alignItems: "center", gap: "16px",
        }}>
          {/* Logo — click to go back to landing page */}
          <div onClick={() => navigate("/")}
            style={{ display: "flex", alignItems: "center", gap: "9px", flexShrink: 0, cursor: "pointer" }}>
            {COMPASS_SVG}
            <span style={{ fontSize: "17px", fontWeight: 700, color: "#fff", letterSpacing: "-0.01em" }}>
              StreamCompass
            </span>
          </div>

          <div style={{ flex: 1 }} />

          {/* Search bar — wider, with a cyan arrow submit button on the right.
               Platform chips have moved to the filter bar below the results headline
               for consistency with the other filters. */}
          <form onSubmit={handleSearchSubmit}
            style={{ flexShrink: 0, width: "440px", display: "flex", gap: "8px", alignItems: "center" }}>
            <div style={{ position: "relative", flex: 1 }}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
                style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}>
                <circle cx="6" cy="6" r="4.5" stroke="#999" strokeWidth="1.2" />
                <line x1="9.5" y1="9.5" x2="13" y2="13" stroke="#999" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
              <input type="text" placeholder="Search titles…" value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                style={{
                  width: "100%", background: "#0d0d0d", border: "1px solid #1e1e1e",
                  borderRadius: "8px", color: "#fff", fontSize: "13px",
                  padding: "10px 12px 10px 34px", outline: "none",
                  fontFamily: "inherit", boxSizing: "border-box", transition: "border-color 0.15s",
                }}
                onFocus={(e) => (e.target.style.borderColor = "#444")}
                onBlur={(e) => (e.target.style.borderColor = "#1e1e1e")}
              />
            </div>
            {/* Arrow button — turns cyan when input has a value, matches landing page CTA */}
            <button type="submit"
              style={{
                flexShrink: 0,
                background: inputValue.trim() ? "#00E5FF" : "#111",
                border: `1px solid ${inputValue.trim() ? "#00E5FF" : "#444"}`,
                borderRadius: "8px",
                color: inputValue.trim() ? "#000" : "#999",
                width: "40px", height: "40px",
                cursor: inputValue.trim() ? "pointer" : "default",
                display: "flex", alignItems: "center", justifyContent: "center",
                transition: "all 0.15s",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <line x1="2" y1="7" x2="11" y2="7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                <polyline points="7,3 11,7 7,11" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </form>
        </div>
      </header>

      {/* ── Hero / results context + filter bar ── */}
      <div style={{ maxWidth: "1600px", margin: "0 auto", padding: "48px 65px 24px" }}>
        <p style={{ fontSize: "12px", color: "#999", letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 600, marginBottom: "8px" }}>
          ML-powered recommendations
        </p>
        {query ? (
          <>
            <h1 style={{ fontSize: "32px", fontWeight: 700, color: "#fff", margin: 0, letterSpacing: "-0.02em", lineHeight: 1.1 }}>
              Results for <span style={{ color: "#00E5FF" }}>"{query}"</span>
            </h1>
            <p style={{ fontSize: "14px", color: "#999", marginTop: "10px", marginBottom: 0 }}>
              {results.length} titles found · Ranked by similarity
            </p>

            {/* Filter bar — shown only when there are results.
                 Row 1: streaming platform chips (multi-select, moved from nav).
                 Row 2: content type + min IMDb toggles. */}
            <div style={{ marginTop: "18px", display: "flex", flexDirection: "column", gap: "12px" }}>

              {/* ── Row 1: Platform chips ── */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                <span style={{ fontSize: "12px", color: "#999", fontFamily: "inherit", whiteSpace: "nowrap" }}>Platform</span>
                <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                  {PLATFORM_FILTER_OPTIONS.filter((p) => p !== "All").map((p) => {
                    const key = PLATFORM_KEY_MAP[p];
                    const isActive = activePlatforms.has(key);
                    return (
                      <button key={p} onClick={() => togglePlatform(key)}
                        style={{
                          background: isActive ? "#fff" : "transparent",
                          border: `1px solid ${isActive ? "#fff" : "#444"}`,
                          borderRadius: "20px",
                          color: isActive ? "#000" : "#999",
                          fontSize: "12px", fontWeight: isActive ? 600 : 400,
                          padding: "4px 14px", cursor: "pointer",
                          whiteSpace: "nowrap", transition: "all 0.15s", fontFamily: "inherit",
                        }}
                      >{p}</button>
                    );
                  })}
                  {activePlatforms.size > 0 && (
                    <button onClick={() => setActivePlatforms(new Set())}
                      style={{
                        background: "transparent", border: "none", color: "#00E5FF",
                        fontSize: "12px", cursor: "pointer", fontFamily: "inherit", padding: "4px 0",
                      }}
                    >✕ Clear</button>
                  )}
                </div>
              </div>

              {/* ── Row 2: Type + Min IMDb ── */}
              <div style={{ display: "flex", alignItems: "center", gap: "20px", flexWrap: "wrap" }}>

                {/* Content type */}
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "12px", color: "#999", fontFamily: "inherit", whiteSpace: "nowrap" }}>Type</span>
                  <div style={{ display: "flex", gap: "5px" }}>
                    {["All", "Movies", "TV Shows"].map((opt) => (
                      <button key={opt} onClick={() => setContentTypeFilter(opt)}
                        style={{
                          background: contentTypeFilter === opt ? "#1a1a1a" : "transparent",
                          border: `1px solid ${contentTypeFilter === opt ? "#999" : "#444"}`,
                          borderRadius: "6px", color: contentTypeFilter === opt ? "#fff" : "#999",
                          fontSize: "12px", padding: "4px 10px", cursor: "pointer",
                          fontFamily: "inherit", transition: "all 0.15s", whiteSpace: "nowrap",
                        }}
                      >{opt}</button>
                    ))}
                  </div>
                </div>

                <div style={{ width: "1px", height: "20px", background: "#444" }} />

                {/* Min IMDb */}
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "12px", color: "#999", fontFamily: "inherit", whiteSpace: "nowrap" }}>Min IMDb</span>
                  <div style={{ display: "flex", gap: "5px" }}>
                    {["Any", "6+", "7+", "8+", "9+"].map((opt) => (
                      <button key={opt} onClick={() => setMinImdbFilter(opt)}
                        style={{
                          background: minImdbFilter === opt ? "#1a1a1a" : "transparent",
                          border: `1px solid ${minImdbFilter === opt ? "#999" : "#444"}`,
                          borderRadius: "6px", color: minImdbFilter === opt ? "#fff" : "#999",
                          fontSize: "12px", padding: "4px 10px", cursor: "pointer",
                          fontFamily: "inherit", transition: "all 0.15s", whiteSpace: "nowrap",
                        }}
                      >{opt}</button>
                    ))}
                  </div>
                </div>

                {(contentTypeFilter !== "All" || minImdbFilter !== "Any") && (
                  <button onClick={() => { setContentTypeFilter("All"); setMinImdbFilter("Any"); }}
                    style={{
                      background: "transparent", border: "none", color: "#00E5FF",
                      fontSize: "12px", cursor: "pointer", fontFamily: "inherit", padding: 0,
                    }}
                  >Clear filters ×</button>
                )}
              </div>
            </div>
          </>
        ) : (
          <>
            <h1 style={{ fontSize: "36px", fontWeight: 700, color: "#fff", margin: 0, letterSpacing: "-0.02em", lineHeight: 1.1 }}>
              What are you watching <span style={{ color: "#00E5FF" }}>next?</span>
            </h1>
            <p style={{ fontSize: "14px", color: "#999", marginTop: "10px", marginBottom: 0 }}>
              Search a title above to get recommendations
            </p>
          </>
        )}
      </div>

      {/* ── Result rows ── */}
      <main style={{
        maxWidth: "1600px", margin: "0 auto",
        padding: "8px 65px",
        paddingBottom: selected ? "340px" : "64px",
      }}>
        {query ? (
          rows.map((row) => (
            <ScrollRow
              key={row.label}
              label={row.label}
              items={row.items}
              onSelect={handleSelect}
              activeId={selected?.title}
            />
          ))
        ) : (
          <div style={{ textAlign: "center", padding: "80px 0", color: "#444" }}>
            <p style={{ fontSize: "14px", fontFamily: "inherit" }}>
              Use the search bar above or go back to the{" "}
              <span onClick={() => navigate("/")} style={{ color: "#00E5FF", cursor: "pointer" }}>
                home page
              </span>
            </p>
          </div>
        )}
      </main>

      <DetailDrawer item={selected} onClose={() => setSelected(null)} />
    </div>
  );
}

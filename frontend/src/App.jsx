/**
 * App.jsx — StreamCompass browse / results page
 *
 * Rendered at "/browse". Reads the ?q= URL param set by LandingPage,
 * fires the real backend recommendation pipeline, and displays results
 * in horizontal scrollable rows grouped by type and quality.
 *
 * API endpoints used:
 *   POST /recommend  — main ML recommendation pipeline (8-stage, see recommender.py)
 *                      Request body: { query, top_k, lambda_div, platforms?, content_type?, min_imdb? }
 *   GET  /title/{t}  — full metadata for a single title (description, cast, director)
 *
 * External dependencies:
 *   react-router-dom — useSearchParams (read ?q=), useNavigate (logo → home)
 */

import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { useNavigate, useSearchParams } from "react-router-dom";

/** Base URL for all backend API calls. */
const API_BASE = "https://api.khayrul.com";

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

/** Platform filter chip labels shown in the filter bar. */
const PLATFORM_FILTER_OPTIONS = ["All", "Netflix", "Hulu", "Prime Video", "Disney+"];

/**
 * Maps human-readable chip label to the platform key used in the API request.
 * Passed directly as the `platforms` array in POST /recommend body.
 */
const PLATFORM_KEY_MAP = {
  "Netflix":     "netflix",
  "Hulu":        "hulu",
  "Prime Video": "prime_video",
  "Disney+":     "disney_plus",
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Extracts which streaming platforms a result item is available on.
 * The API returns binary flags (0 or 1) per platform column.
 *
 * @param {Object} item - A result object from POST /recommend
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
 * MatchTooltip — breakdown popup shown when hovering the similarity % badge.
 *
 * Reads `match_breakdown` from the result object, populated by the backend's
 * recommender.py weighted TF-IDF scoring. Each field (description, genres,
 * cast, director) has its own similarity score shown as a bar.
 *
 * @param {Object} breakdown - item.match_breakdown from POST /recommend
 */
// App font stack — set explicitly on the tooltip because it renders through a
// portal into document.body, outside the root div, so "inherit" would fall
// back to the browser default serif font instead of the app font.
const APP_FONT = "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif";

function MatchTooltip({ breakdown }) {
  if (!breakdown) {
    return (
      <div style={{
        width: "420px", background: "#0d0d0d", border: "1px solid #333",
        borderRadius: "8px", padding: "14px 16px",
        boxSizing: "border-box", zIndex: 200,
        display: "flex", alignItems: "center", justifyContent: "center",
        boxShadow: "0 8px 24px rgba(0,0,0,0.6)", fontFamily: APP_FONT,
      }}>
        <p style={{ fontSize: "11px", color: "#555", margin: 0, fontFamily: "inherit" }}>
          Match breakdown not available
        </p>
      </div>
    );
  }

  // All values from item.match_breakdown, with ?? 0 fallbacks.
  const finalScore       = breakdown.final_score ?? 0;
  const contentSimilarity = breakdown.content_similarity ?? 0;
  const imdb             = breakdown.imdb ?? 0;
  const popularity       = breakdown.popularity ?? 0;
  const contentWeight    = breakdown.content_weight ?? 0;
  const imdbWeight       = breakdown.imdb_weight ?? 0;
  const popularityWeight = breakdown.popularity_weight ?? 0;

  const sectionHeading = {
    fontSize: "10px", color: "#888", fontWeight: 700,
    letterSpacing: "0.07em", textTransform: "uppercase",
    fontFamily: "inherit",
  };

  // Left column: per-signal contribution rows (weight × value).
  const contentContribution    = contentWeight * contentSimilarity;
  const imdbContribution        = imdbWeight * imdb;
  const popularityContribution  = popularityWeight * popularity;
  const linearTotal             = contentContribution + imdbContribution + popularityContribution;

  const scoreRows = [
    {
      label: "Content",
      contribution: contentContribution,
      subtitle: `weight × similarity = ${contentWeight} × ${contentSimilarity.toFixed(2)}`,
    },
    {
      label: "IMDb",
      contribution: imdbContribution,
      subtitle: `weight × score = ${imdbWeight} × ${imdb.toFixed(2)}`,
    },
    {
      label: "Popularity",
      contribution: popularityContribution,
      subtitle: `weight × rank = ${popularityWeight} × ${popularity.toFixed(2)}`,
    },
  ];

  // Right column: raw TF-IDF field similarity bars.
  const fields = [
    { key: "description", label: "Description" },
    { key: "genres",      label: "Genres" },
    { key: "cast",        label: "Cast" },
    { key: "director",    label: "Director" },
  ];

  return (
    <div style={{
      width: "420px", background: "#0d0d0d", border: "1px solid #333",
      borderRadius: "8px", padding: "14px 16px",
      display: "flex", gap: "16px", alignItems: "stretch",
      overflow: "hidden", boxSizing: "border-box",
      boxShadow: "0 8px 24px rgba(0,0,0,0.6)", fontFamily: APP_FONT,
    }}
      onClick={(e) => e.stopPropagation()}
    >
      {/* ── Left column: final score + contribution breakdown ── */}
      <div style={{
        flex: "0 0 190px", borderRight: "1px solid #2a2a2a", paddingRight: "16px",
        display: "flex", flexDirection: "column",
      }}>
        {/* Top block — final score */}
        <div>
          <div style={{ ...sectionHeading, marginBottom: "4px" }}>Final Score</div>
          <div style={{ fontSize: "24px", color: "#00E5FF", fontWeight: 700, fontFamily: "inherit", lineHeight: 1, marginBottom: "4px" }}>
            {Math.round(finalScore * 100)}%
          </div>
          <div style={{ fontSize: "9px", color: "#888", fontStyle: "italic", fontFamily: "inherit" }}>
            refined by neural network scoring
          </div>
        </div>

        {/* Divider */}
        <div style={{ borderTop: "1px solid #2a2a2a", margin: "10px 0" }} />

        {/* Bottom block — component contributions + linear total */}
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", flex: 1, gap: "8px" }}>
          {scoreRows.map(({ label, contribution, subtitle }) => (
            <div key={label}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span style={{ fontSize: "10px", color: "#ccc", fontWeight: 600, fontFamily: "inherit" }}>{label}</span>
                <span style={{ fontSize: "10px", color: "#00E5FF", fontWeight: 700, fontFamily: "inherit" }}>
                  {(contribution * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ fontSize: "9px", color: "#888", fontFamily: "inherit", marginTop: "1px" }}>
                {subtitle}
              </div>
            </div>
          ))}

          {/* Linear total */}
          <div style={{ borderTop: "1px solid #2a2a2a", paddingTop: "6px", display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span style={{ fontSize: "9px", color: "#888", fontFamily: "inherit" }}>Linear total</span>
            <span style={{ fontSize: "9px", color: "#00E5FF", fontWeight: 700, fontFamily: "inherit" }}>
              {(linearTotal * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* ── Right column: content breakdown bars ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <div style={{ ...sectionHeading, marginBottom: "8px" }}>Content Breakdown</div>

        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", flex: 1 }}>
          {fields.map(({ key, label }) => {
            const raw = breakdown[key] ?? 0;
            const barWidth = Math.round(raw * 100);
            const labelColor = barWidth > 60 ? "#00E5FF" : barWidth > 30 ? "#4dd0e1" : "#4dd0e1";
            const fillColor = barWidth > 60 ? "#00E5FF" : barWidth > 30 ? "#4dd0e1" : "#1e6b78";
            return (
              <div key={key}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "3px" }}>
                  <span style={{ fontSize: "10px", color: "#ccc", fontFamily: "inherit" }}>{label}</span>
                  <span style={{ fontSize: "10px", color: labelColor, fontWeight: 600, fontFamily: "inherit" }}>{barWidth}%</span>
                </div>
                <div style={{ height: "7px", background: "#1e1e1e", borderRadius: "3px", overflow: "hidden" }}>
                  <div style={{
                    height: "100%",
                    width: `${Math.min(100, barWidth)}%`,
                    background: fillColor,
                    borderRadius: "3px",
                    transition: "width 0.3s ease",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
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
 * @param {Object}   item     - Result object from POST /recommend
 * @param {Function} onClick  - Called with item when card is clicked
 * @param {boolean}  isActive - Whether this card's drawer is currently open
 */
function TitleCard({ item, onClick, isActive }) {
  const [hovered, setHovered] = useState(false);
  const [badgeHovered, setBadgeHovered] = useState(false);
  const cardRef = useRef(null);
  const [tooltipPos, setTooltipPos] = useState(null);
  const initials = item.title.split(" ").slice(0, 2).map((w) => w[0]).join("").toUpperCase();
  const simPct = item.similarity_score ? Math.round(item.similarity_score * 100) : null;

  // Anchor the tooltip above the card centre in viewport (fixed) coordinates so
  // it escapes the horizontal scroll row's overflow clipping.
  const showTooltip = () => {
    const rect = cardRef.current?.getBoundingClientRect();
    if (rect) {
      // Tooltip is ~460px wide; clamp its centre so the panel stays on-screen
      // for cards near the left/right viewport edges.
      const halfWidth = 234;
      const center = rect.left + rect.width / 2;
      const clamped = Math.min(
        Math.max(center, halfWidth + 8),
        window.innerWidth - halfWidth - 8
      );
      setTooltipPos({
        left: clamped,
        bottom: window.innerHeight - rect.top + 8, // 8px gap above the card
      });
    }
    setBadgeHovered(true);
  };

  return (
    <div
      ref={cardRef}
      onClick={() => onClick(item)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setBadgeHovered(false); }}
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
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: "#00E5FF", opacity: 0.8, zIndex: 1 }} />
        )}

        {/* Poster image — shown when backend provides poster_url (TMDB).
             Falls back to monogram initials until poster_url is available. */}
        {item.poster_url ? (
          <img
            src={item.poster_url}
            alt={item.title}
            style={{
              position: "absolute", top: 0, left: 0,
              width: "100%", height: "100%",
              objectFit: "cover", borderRadius: "6px",
            }}
            onError={(e) => { e.target.style.display = "none"; }}
          />
        ) : (
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
        )}

        {/* Title + year overlay at the bottom */}
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          padding: "12px 10px", zIndex: 1,
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

      </div>

      {/* Similarity badge — positioned over the top-right corner of the card face.
           The outer wrapper is relative and sits between the card div and genre label,
           with negative margin to pull it up into the card. The tooltip drops DOWN
           from the badge, overlaying the card face. overflow:visible so it escapes. */}
      {simPct >= 80 && (
        <div
          style={{ position: "relative", height: 0, overflow: "visible" }}
          onMouseEnter={(e) => { e.stopPropagation(); showTooltip(); }}
          onMouseLeave={(e) => { e.stopPropagation(); setBadgeHovered(false); }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Badge pill — top-right corner of the card, same as original */}
          <div style={{
            position: "absolute", top: "-202px", right: "8px",
            background: badgeHovered ? "rgba(0,229,255,0.25)" : "rgba(0,229,255,0.15)",
            border: `1px solid ${badgeHovered ? "rgba(0,229,255,0.8)" : "rgba(0,229,255,0.4)"}`,
            borderRadius: "3px", padding: "2px 5px",
            fontSize: "9px", fontWeight: 700, color: "#00E5FF",
            fontFamily: "inherit", letterSpacing: "0.05em",
            cursor: "default", transition: "all 0.15s",
            userSelect: "none", zIndex: 20,
          }}>
            {simPct}%
          </div>

          {/* Tooltip — rendered into a body portal with fixed positioning so it
               pops UP above the card and is never clipped by the scroll row. */}
          {badgeHovered && tooltipPos && createPortal(
            <div style={{
              position: "fixed",
              left: `${tooltipPos.left}px`,
              bottom: `${tooltipPos.bottom}px`,
              transform: "translateX(-50%)",
              zIndex: 1000,
            }}>
              <MatchTooltip breakdown={item.match_breakdown} />
            </div>,
            document.body
          )}
        </div>
      )}

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
 * Shows skeleton placeholder cards while `loading` is true.
 *
 * @param {string}   label    - Row heading
 * @param {Array}    items    - Cards to render
 * @param {Function} onSelect - Passed to each TitleCard onClick
 * @param {string}   activeId - Title of the currently open drawer card
 * @param {boolean}  loading  - When true, renders skeleton placeholders
 */
function ScrollRow({ label, items, onSelect, activeId, loading }) {
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

      {/* Skeleton loading placeholders */}
      {loading ? (
        <div style={{ display: "flex", gap: "12px" }}>
          {[...Array(6)].map((_, i) => (
            <div key={i} style={{
              flexShrink: 0, width: "148px", height: "210px", borderRadius: "6px",
              background: "#0d0d0d", border: "1px solid #1a1a1a",
              animation: "pulse 1.5s ease-in-out infinite",
              animationDelay: `${i * 0.1}s`,
            }} />
          ))}
        </div>
      ) : items.length === 0 ? (
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
 * DrawerMatchBadge — inline hoverable % match pill for the DetailDrawer.
 * Reuses MatchTooltip for the breakdown, same as the card badge.
 * Positioned with relative so the tooltip flows naturally in the drawer.
 *
 * @param {Object} item - The selected title object (has similarity_score + match_breakdown)
 */
function DrawerMatchBadge({ item }) {
  const [hovered, setHovered] = useState(false);
  const simPct = Math.round(item.similarity_score * 100);

  return (
    <div
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <span style={{
        fontSize: "11px", color: "#00E5FF", fontFamily: "inherit",
        cursor: "default",
        borderBottom: hovered ? "1px dashed rgba(0,229,255,0.5)" : "1px dashed transparent",
        transition: "border-color 0.15s",
        userSelect: "none",
      }}>
        {simPct}% match ↗
      </span>
      {hovered && (
        <div style={{ position: "absolute", bottom: "calc(100% + 6px)", left: 0, zIndex: 200 }}>
          <MatchTooltip breakdown={item.match_breakdown} />
        </div>
      )}
    </div>
  );
}

/**
 * DetailDrawer — slide-up panel showing full metadata for a selected title.
 *
 * Returns null when no item is selected (zero DOM footprint when closed).
 * On card click, the parent fetches GET /title/{name} for full detail
 * (description, cast, director) and merges it onto the base result data.
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
        @keyframes pulse   { 0%,100% { opacity: 0.4; } 50% { opacity: 0.8; } }
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
              {item.year && <span style={{ fontSize: "13px", color: "#444", fontFamily: "inherit" }}>{Math.round(item.year)}</span>}
              <span style={{ fontSize: "13px", color: "#444" }}>·</span>
              <span style={{ fontSize: "11px", border: "1px solid #444", borderRadius: "3px", padding: "1px 6px", color: "#888", fontFamily: "inherit" }}>
                {item.content_type === "tv" ? "TV Show" : "Movie"}
              </span>
              <StarRating rating={item.imdb_score} />
              {/* % match — hoverable, shows same MatchTooltip breakdown as the card badge */}
              {item.similarity_score && (
                <DrawerMatchBadge item={item} />
              )}
            </div>
          </div>

          {/* Description — present when detail was fetched from GET /title */}
          {item.description && (
            <p style={{ fontSize: "13px", color: "#aaa", lineHeight: 1.65, margin: "0 0 14px", fontFamily: "inherit" }}>
              {item.description}
            </p>
          )}

          {/* Cast — present in full detail response */}
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
 *   query             {string}      - Active search term (from URL param or nav input)
 *   inputValue        {string}      - Live value in the nav search input
 *   selected          {Object|null} - Last-clicked card (partial data from /recommend)
 *   titleDetail       {Object|null} - Full detail merged from GET /title
 *   activePlatforms   {Set}         - Platform keys currently toggled on
 *   contentTypeFilter {string}      - "All" | "Movies" | "TV Shows"
 *   minImdbFilter     {string}      - "Any" | "6+" | "7+" | "8+" | "9+"
 *   results           {Array}       - Raw results from POST /recommend
 *   loading           {boolean}     - True while /recommend is in-flight
 *   error             {string|null} - Error message if fetch fails
 */
export default function App() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const initialQuery = searchParams.get("q") || "";

  const [query, setQuery]           = useState(initialQuery);
  const [inputValue, setInputValue] = useState(initialQuery);
  const [selected, setSelected]     = useState(null);
  const [titleDetail, setTitleDetail] = useState(null);
  const [results, setResults]       = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);

  /**
   * activePlatforms — Set of platform keys currently selected by the user.
   * Empty set means no platform filter. Multiple platforms can be active
   * simultaneously — clicking a chip toggles it on/off.
   * Passed as the `platforms` array in POST /recommend body.
   */
  const [activePlatforms, setActivePlatforms] = useState(new Set());

  /**
   * Toggles a platform chip on or off.
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
   * Content type filter — passed as `content_type` in POST /recommend body.
   * "All" means no content_type filter is sent.
   */
  const [contentTypeFilter, setContentTypeFilter] = useState("All");

  /**
   * Min IMDb filter — passed as `min_imdb` in POST /recommend body.
   * "Any" means no min_imdb filter is sent.
   */
  const [minImdbFilter, setMinImdbFilter] = useState("Any");

  /**
   * Main recommendation fetch effect.
   * Re-runs whenever query, activePlatforms, contentTypeFilter, or minImdbFilter changes.
   *
   * Sends POST /recommend with all active filters.
   * On success: populates `results`.
   * On failure: sets `error`.
   */
  useEffect(() => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setSelected(null);
    setTitleDetail(null);

    // Build request body — only include optional filters when active
    const body = {
      query: query.trim(),
      top_k: 40,
      lambda_div: 0.3,
      ...(activePlatforms.size > 0 ? { platforms: [...activePlatforms] } : {}),
      ...(contentTypeFilter === "Movies" ? { content_type: "movie" } : {}),
      ...(contentTypeFilter === "TV Shows" ? { content_type: "tv" } : {}),
      ...(minImdbFilter !== "Any" ? { min_imdb: parseFloat(minImdbFilter) } : {}),
    };

    fetch(`${API_BASE}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => r.json())
      .then((data) => {
        setResults(data.results || []);
        setLoading(false);
      })
      .catch(() => {
        setError("Couldn't reach the server. Check your connection and try again.");
        setLoading(false);
      });
  }, [query, activePlatforms, contentTypeFilter, minImdbFilter]);

  /**
   * Handles card click.
   * Clicking the same card again closes the drawer (toggle).
   * Clicking a new card:
   *   1. Sets `selected` immediately (drawer opens with partial data)
   *   2. Fetches GET /title/{name} for full metadata
   *   3. Merges full detail onto the partial result via `titleDetail`
   * Falls back to partial data if the /title fetch fails.
   */
  const handleSelect = async (item) => {
    if (selected?.title === item.title) {
      setSelected(null);
      setTitleDetail(null);
      return;
    }
    setSelected(item);
    setTitleDetail(null);

    try {
      const res = await fetch(`${API_BASE}/title/${encodeURIComponent(item.title)}`);
      if (res.ok) {
        const detail = await res.json();
        // Keep similarity_score from /recommend, add description/cast/director from /title
        setTitleDetail({ ...item, ...detail });
      } else {
        setTitleDetail(item);
      }
    } catch {
      setTitleDetail(item); // fallback to partial data on error
    }
  };

  /**
   * Handles nav bar search form submission.
   * Updates both the URL param (for shareability) and the `query` state
   * that triggers the /recommend effect above.
   */
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    setQuery(inputValue.trim());
    navigate(`/browse?q=${encodeURIComponent(inputValue.trim())}`, { replace: true });
  };

  // Derive row data by splitting flat results into themed groups
  const topMatches   = results.slice(0, 15);
  const movieResults = results.filter((r) => r.content_type === "movie");
  const tvResults    = results.filter((r) => r.content_type === "tv");
  const highRated    = results.filter((r) => r.imdb_score && r.imdb_score >= 8.0);

  const rows = query ? [
    { label: `Best matches for "${query}"`,    items: topMatches   },
    ...(contentTypeFilter === "All" && movieResults.length > 0 ? [{ label: "Movies",              items: movieResults }] : []),
    ...(contentTypeFilter === "All" && tvResults.length    > 0 ? [{ label: "TV Shows",            items: tvResults    }] : []),
    ...(highRated.length    > 0                               ? [{ label: "Highly rated picks",   items: highRated    }] : []),
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

          {/* Search bar — wider, with cyan arrow submit button.
               Platform chips are in the filter bar below for consistency. */}
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
            {/* Arrow button — turns cyan when input has a value */}
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
              Results for <span style={{ color: "#00E5FF" }}>&quot;{query}&quot;</span>
            </h1>
            <p style={{ fontSize: "14px", color: "#999", marginTop: "10px", marginBottom: 0 }}>
              {loading ? "Finding recommendations…" : `${results.length} titles found · Ranked by similarity`}
            </p>

            {/* Filter bar — Row 1: platforms, Row 2: type + IMDb */}
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

            {/* Error banner */}
            {error && (
              <p style={{ fontSize: "13px", color: "#c0392b", marginTop: "16px", background: "#1a0a0a", border: "1px solid #3a1a1a", borderRadius: "6px", padding: "10px 14px" }}>
                {error}
              </p>
            )}
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
              loading={loading}
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

      {/* Detail drawer — fixed at bottom, slides up on card click */}
      <DetailDrawer
        item={titleDetail || selected}
        onClose={() => { setSelected(null); setTitleDetail(null); }}
      />
    </div>
  );
}

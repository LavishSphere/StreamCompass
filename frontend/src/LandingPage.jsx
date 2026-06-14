/**
 * LandingPage.jsx — StreamCompass entry point
 *
 * This is the first page users see at "/". It renders a minimal, centered
 * search interface — just the logo, a headline, and a search bar.
 *
 * Flow:
 *   1. User types a title (e.g. "Breaking Bad")
 *   2. As they type, we debounce-fetch GET /search for autocomplete suggestions
 *   3. On Enter or suggestion click, we navigate to /browse?q=<query>
 *   4. The browse page (App.jsx) picks up the ?q= param and fires POST /recommend
 *
 * API endpoints used:
 *   GET /search?q={query}&limit=6  — autocomplete suggestions while typing
 *
 * External dependencies:
 *   react-router-dom  — useNavigate for client-side navigation to /browse
 */

import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

/** Base URL for all backend API calls. */
const API_BASE = "https://api.khayrul.com";

/**
 * Renders the StreamCompass compass SVG logo at a given pixel size.
 * Extracted as a function so it can be reused at different sizes
 * (44px on landing, 26px in the nav bar).
 *
 * @param {number} size - Width and height in pixels
 * @returns {JSX.Element} SVG element
 */
const CompassSVG = ({ size = 32 }) => (
  <svg width={size} height={size} viewBox="0 0 727 727" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M726.286 382.265C716.048 579.042 550.857 731.808 353.88 726.652C156.909 721.499 -0.0698985 560.3 2.33495e-05 363.261C0.0699452 166.22 157.163 5.13451 354.138 0.119903C551.118 -4.89471 716.204 147.989 726.298 344.771C726.939 357.261 726.935 369.776 726.286 382.265ZM363.04 648.241C437.939 647.726 509.575 617.523 562.228 564.257C626.165 500.534 660.884 408.214 643.736 318.289C632.208 257.853 606.505 206.438 562.747 162.732C512.333 112.378 443.751 79.9413 371.876 80.0242C368.845 80.0203 365.81 80.0499 362.775 80.1128C346.039 80.7902 329.937 81.1175 313.342 83.8046C177.884 105.735 73.5098 241.516 78.3039 377.081C79.3363 406.276 85.8055 434.276 96.9274 461.101C113.551 504.499 136.274 538.394 170.763 569.976C223.302 618.429 291.601 646.23 363.04 648.241Z" fill="#00E5FF"/>
    <path d="M534.263 212.763L534.611 213.28C526.431 230.554 517.181 248.382 508.595 265.552L460.814 361.089C456.775 369.171 428.458 427.593 425.599 430.241C422.38 433.226 386.044 450.441 379.392 453.765L228.623 529.163C222.233 532.351 215.075 535.679 208.859 539.058L208.58 538.527L303.188 349.061C307.608 340.207 312.584 329.628 317.31 321.093L534.263 212.763ZM250.367 497.003C265.782 488.796 282.191 480.956 297.845 473.042L357.111 443.417C367.431 438.249 378.552 432.417 388.962 427.585C383.759 421.913 377.482 415.894 372.001 410.405L342.477 380.862C339.178 377.562 322.682 360.253 319.631 358.554C314.848 368.675 309.354 379.226 304.318 389.273L278.589 440.663C269.598 458.542 259.788 479.71 250.367 497.003Z" fill="#00E5FF"/>
  </svg>
);

/**
 * Small colored platform identifier badge shown in autocomplete suggestions.
 * Each streaming service gets its brand color so users can instantly see
 * where a title is available before navigating to the full results page.
 *
 * @param {string} color - Hex background color (brand color)
 * @param {string} label - Short label text (e.g. "N", "H", "D+")
 */
function PlatformDot({ color, label }) {
  return (
    <span style={{
      background: color,
      color: "#fff",
      fontSize: "9px",
      fontWeight: 700,
      padding: "2px 5px",
      borderRadius: "3px",
      fontFamily: "inherit",
      letterSpacing: "0.03em",
    }}>
      {label}
    </span>
  );
}

/**
 * LandingPage — main export
 *
 * State:
 *   query              {string}  - Current text in the search input
 *   suggestions        {Array}   - Autocomplete results from GET /search
 *   showSuggestions    {boolean} - Whether the dropdown is visible
 *   loadingSuggestions {boolean} - Spinner shown while /search is in-flight
 *   activeSuggestion   {number}  - Keyboard-highlighted suggestion index (-1 = none)
 *
 * Refs:
 *   inputRef    - Direct DOM ref to the <input> so we can auto-focus on mount
 *   debounceRef - Stores the setTimeout ID so we can cancel it on each keystroke
 */
export default function LandingPage() {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);

  const inputRef = useRef(null);
  const debounceRef = useRef(null);
  const navigate = useNavigate();

  /** Auto-focus the search input when the landing page first mounts. */
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  /**
   * Autocomplete effect — fires whenever `query` changes.
   *
   * Debounced by 280ms so we don't hammer the API on every keystroke.
   * Skips the fetch if the query is fewer than 2 characters.
   * Calls GET /search which returns titles whose normalised name contains
   * the query string, sorted by exact-start matches first then IMDb score.
   */
  useEffect(() => {
    if (!query.trim() || query.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    // Cancel any pending debounce from the previous keystroke
    clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
      setLoadingSuggestions(true);
      try {
        const res = await fetch(
          `${API_BASE}/search?q=${encodeURIComponent(query)}&limit=6`
        );
        const data = await res.json();
        setSuggestions(data.results || []);
        setShowSuggestions(true);
        setActiveSuggestion(-1);
      } catch {
        // Silently fail — autocomplete is a nice-to-have, not critical
        setSuggestions([]);
      } finally {
        setLoadingSuggestions(false);
      }
    }, 280);

    // Cleanup: cancel the debounce if the component unmounts mid-wait
    return () => clearTimeout(debounceRef.current);
  }, [query]);

  /**
   * Navigates to the browse page with the search query as a URL param.
   * App.jsx reads this param on mount and fires the /recommend call.
   *
   * @param {string} [searchQuery] - Optional override; uses `query` state if omitted
   */
  const handleSearch = (searchQuery) => {
    const q = (searchQuery || query).trim();
    if (!q) return;
    setShowSuggestions(false);
    navigate(`/browse?q=${encodeURIComponent(q)}`);
  };

  /**
   * Keyboard navigation for the autocomplete dropdown.
   *   ArrowDown / ArrowUp — move the highlighted suggestion
   *   Enter               — submit highlighted suggestion or raw query
   *   Escape              — close the dropdown
   */
  const handleKeyDown = (e) => {
    if (!showSuggestions || suggestions.length === 0) {
      if (e.key === "Enter") handleSearch();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveSuggestion((prev) => Math.min(prev + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveSuggestion((prev) => Math.max(prev - 1, -1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeSuggestion >= 0) {
        handleSearch(suggestions[activeSuggestion].title);
      } else {
        handleSearch();
      }
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
      setActiveSuggestion(-1);
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#000",
      color: "#fff",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "0 24px",
    }}>

      {/* ── Logo ── */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "48px" }}>
        <CompassSVG size={44} />
        <span style={{ fontSize: "28px", fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>
          StreamCompass
        </span>
      </div>

      {/* ── Headline ── */}
      <h1 style={{
        fontSize: "clamp(28px, 5vw, 52px)",
        fontWeight: 700,
        color: "#fff",
        margin: "0 0 12px",
        letterSpacing: "-0.03em",
        textAlign: "center",
        lineHeight: 1.1,
      }}>
        What are you watching{" "}
        <span style={{ color: "#00E5FF" }}>next?</span>
      </h1>
      <p style={{ fontSize: "16px", color: "#999", marginBottom: "40px", textAlign: "center" }}>
        Search any title and find what to watch next across various streaming platforms
      </p>

      {/* ── Search box + autocomplete dropdown ── */}
      <div style={{ width: "100%", maxWidth: "580px", position: "relative" }}>

        {/* Input wrapper — styled as a single pill */}
        <div style={{
          display: "flex",
          alignItems: "center",
          background: "#0d0d0d",
          border: "1px solid #2a2a2a",
          borderRadius: "12px",
          padding: "0 16px",
          transition: "border-color 0.15s",
        }}>
          {/* Search icon */}
          <svg width="18" height="18" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }}>
            <circle cx="6" cy="6" r="4.5" stroke="#444" strokeWidth="1.2" />
            <line x1="9.5" y1="9.5" x2="13" y2="13" stroke="#444" strokeWidth="1.2" strokeLinecap="round" />
          </svg>

          <input
            ref={inputRef}
            type="text"
            placeholder="Search a title, e.g. Breaking Bad, Inception…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={(e) => {
              e.currentTarget.closest("div").style.borderColor = "#444";
              if (suggestions.length > 0) setShowSuggestions(true);
            }}
            onBlur={(e) => {
              e.currentTarget.closest("div").style.borderColor = "#2a2a2a";
              // Delay hide so onMouseDown on a suggestion fires before blur closes it
              setTimeout(() => setShowSuggestions(false), 150);
            }}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              color: "#fff",
              fontSize: "16px",
              padding: "18px 12px",
              fontFamily: "inherit",
            }}
          />

          {/* Loading spinner — visible while /search is in-flight */}
          {loadingSuggestions && (
            <div style={{
              width: "16px", height: "16px",
              border: "2px solid #222",
              borderTop: "2px solid #00E5FF",
              borderRadius: "50%",
              animation: "spin 0.7s linear infinite",
              flexShrink: 0,
            }} />
          )}

          {/* Clear button — only shown when there's text and no spinner */}
          {query && !loadingSuggestions && (
            <button
              onClick={() => { setQuery(""); setSuggestions([]); inputRef.current?.focus(); }}
              style={{
                background: "transparent", border: "none", color: "#444",
                cursor: "pointer", padding: "4px", fontSize: "18px",
                lineHeight: 1, flexShrink: 0,
              }}
            >×</button>
          )}
        </div>

        {/* ── Autocomplete dropdown ── */}
        {showSuggestions && suggestions.length > 0 && (
          <div style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            left: 0, right: 0,
            background: "#0d0d0d",
            border: "1px solid #222",
            borderRadius: "10px",
            overflow: "hidden",
            zIndex: 100,
          }}>
            {suggestions.map((s, i) => (
              <div
                key={i}
                onMouseDown={() => handleSearch(s.title)}
                style={{
                  padding: "12px 16px",
                  cursor: "pointer",
                  background: i === activeSuggestion ? "#161616" : "transparent",
                  borderBottom: i < suggestions.length - 1 ? "1px solid #161616" : "none",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "12px",
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#161616")}
                onMouseLeave={(e) => (e.currentTarget.style.background = i === activeSuggestion ? "#161616" : "transparent")}
              >
                {/* Title + year */}
                <div style={{ display: "flex", alignItems: "center", gap: "10px", minWidth: 0 }}>
                  <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }}>
                    <circle cx="6" cy="6" r="4.5" stroke="#444" strokeWidth="1.2" />
                    <line x1="9.5" y1="9.5" x2="13" y2="13" stroke="#444" strokeWidth="1.2" strokeLinecap="round" />
                  </svg>
                  <span style={{ fontSize: "14px", color: "#fff", fontFamily: "inherit", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {s.title}
                  </span>
                  {s.year && (
                    <span style={{ fontSize: "12px", color: "#555", fontFamily: "inherit", flexShrink: 0 }}>
                      {s.year}
                    </span>
                  )}
                </div>

                {/* Platform dots — shows where this title is available */}
                <div style={{ display: "flex", gap: "5px", flexShrink: 0 }}>
                  {s.netflix === 1 && <PlatformDot color="#E50914" label="N" />}
                  {s.hulu === 1 && <PlatformDot color="#1CE783" label="H" />}
                  {s.prime_video === 1 && <PlatformDot color="#00A8E0" label="P" />}
                  {s.disney_plus === 1 && <PlatformDot color="#113CCF" label="D+" />}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Submit button ── */}
      <button
        onClick={() => handleSearch()}
        disabled={!query.trim()}
        style={{
          marginTop: "16px",
          background: query.trim() ? "#00E5FF" : "#111",
          border: "none",
          borderRadius: "8px",
          color: query.trim() ? "#000" : "#333",
          fontSize: "15px",
          fontWeight: 600,
          padding: "12px 32px",
          cursor: query.trim() ? "pointer" : "default",
          transition: "all 0.15s",
          fontFamily: "inherit",
          letterSpacing: "0.01em",
        }}
      >
        Find recommendations →
      </button>

      {/* Dataset size hint */}
      <p style={{ marginTop: "48px", fontSize: "14px", color: "#999", textAlign: "center" }}>
        Searching through 42,945 titles across Netflix, Hulu, Prime Video & Disney+
      </p>

      {/* Keyframe animations injected globally for this page */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

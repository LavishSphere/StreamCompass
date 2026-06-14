/**
 * main.jsx — React application entry point
 *
 * This is the file Vite uses as the root module. It mounts the React app
 * onto the <div id="root"> in index.html and sets up client-side routing
 * via React Router.
 *
 * Routes:
 *   /         → LandingPage  (search-focused entry point)
 *   /browse   → App          (results page, reads ?q= URL param)
 *   *         → redirect to / (any unknown URL goes back to landing)
 *
 * Why BrowserRouter?
 *   Uses the HTML5 History API so URLs look like /browse?q=Inception
 *   rather than /#/browse?q=Inception (HashRouter). Requires the server
 *   to serve index.html for all routes — Vite does this automatically
 *   in dev, and the vite.config.js handles it for the production build.
 *
 * Dependencies:
 *   react-router-dom — install with: npm install react-router-dom
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LandingPage from './LandingPage.jsx'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {/* Landing page — just the search bar */}
        <Route path="/" element={<LandingPage />} />

        {/* Browse/results page — reads ?q= param and calls the backend */}
        <Route path="/browse" element={<App />} />

        {/* Catch-all — redirect any unknown path back to landing */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)

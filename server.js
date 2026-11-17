// Express API Gateway for RELAY Module 6
// Proxies public client requests to the Python Flask microservice (port 5001)

const path = require("node:path");
const express = require("express");

function createApp(aiUrl = process.env.AI_URL || "http://127.0.0.1:5001") {
  const app = express();
  app.use(express.json());

  // Normalize upstream AI URL (auto-prefix http/https if missing, strip trailing slashes)
  let normalizedAiUrl = (aiUrl || "http://127.0.0.1:5001").trim();
  if (!normalizedAiUrl.startsWith("http://") && !normalizedAiUrl.startsWith("https://")) {
    if (normalizedAiUrl.includes(":") || !normalizedAiUrl.includes(".")) {
      normalizedAiUrl = `http://${normalizedAiUrl}`;
    } else {
      normalizedAiUrl = `https://${normalizedAiUrl}`;
    }
  }
  normalizedAiUrl = normalizedAiUrl.replace(/\/+$/, "");

  const aiTimeoutMs = parseInt(process.env.AI_TIMEOUT_MS, 10) || 30000;

  // Static assets for the admin dashboard
  app.use(express.static(path.join(__dirname, "public")));

  // Input validation middleware for vehicleId
  const validateVehicleId = (req, res, next) => {
    const id = (req.params.id || "").trim();
    if (!id || id.length > 50 || !/^[a-zA-Z0-9_-]+$/.test(id)) {
      return res.status(400).json({
        error: "Invalid vehicle ID format. Must be alphanumeric (max 50 chars).",
        providedId: req.params.id
      });
    }
    req.params.id = id;
    next();
  };

  async function proxyAI(req, res, internalPath) {
    try {
      const response = await fetch(`${normalizedAiUrl}${internalPath}`, {
        headers: { "Accept": "application/json" },
        signal: AbortSignal.timeout(aiTimeoutMs)
      });
      const data = await response.json();
      return res.status(response.status).json(data);
    } catch (error) {
      const isTimeout = error.name === "TimeoutError";
      return res.status(isTimeout ? 504 : 502).json({
        error: isTimeout ? "AI service timeout" : "AI service unavailable",
        message: isTimeout
          ? `Request to Flask service timed out after ${aiTimeoutMs}ms.`
          : `Unable to reach Python Flask microservice on ${normalizedAiUrl}`,
        details: "Ensure the Flask service is running on the configured AI_URL."
      });
    }
  }

  // Gateway health check
  app.get("/health", async (_req, res) => {
    try {
      const response = await fetch(`${normalizedAiUrl}/health`, { signal: AbortSignal.timeout(10000) });
      const aiData = await response.json();
      return res.status(200).json({
        status: "ok",
        gateway: "relay-node-gateway",
        upstreamStatus: "connected",
        upstreamUrl: normalizedAiUrl,
        aiService: aiData
      });
    } catch (err) {
      return res.status(503).json({
        status: "degraded",
        gateway: "relay-node-gateway",
        upstreamStatus: "disconnected",
        upstreamUrl: normalizedAiUrl,
        details: "AI service unreachable"
      });
    }
  });

  // Fair Price Endpoint
  app.get("/api/v1/vehicles/:id/price", validateVehicleId, (req, res) =>
    proxyAI(req, res, `/internal/vehicles/${encodeURIComponent(req.params.id)}/price`)
  );

  // Anomaly Score Endpoint
  app.get("/api/v1/vehicles/:id/anomaly-score", validateVehicleId, (req, res) =>
    proxyAI(req, res, `/internal/vehicles/${encodeURIComponent(req.params.id)}/anomaly-score`)
  );

  // Admin Flagged Vehicles API endpoint
  app.get("/api/v1/admin/flagged-vehicles", (req, res) => {
    const filter = req.query.filter ? `?filter=${encodeURIComponent(req.query.filter)}` : "";
    return proxyAI(req, res, `/internal/admin/flagged-vehicles${filter}`);
  });

  // Admin Dashboard Web Page
  app.get("/admin", (_req, res) => {
    res.sendFile(path.join(__dirname, "public", "admin.html"));
  });

  // Root endpoint info
  app.get("/", (_req, res) => res.json({
    name: "RELAY Module 6 API Gateway",
    version: "1.0.0",
    endpoints: [
      "GET /health",
      "GET /admin",
      "GET /api/v1/admin/flagged-vehicles",
      "GET /api/v1/vehicles/:id/price",
      "GET /api/v1/vehicles/:id/anomaly-score"
    ]
  }));

  // 404 handler for undefined routes
  app.use((_req, res) => {
    res.status(404).json({ error: "Endpoint not found" });
  });

  return app;
}

const app = createApp();
const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || "0.0.0.0";

if (require.main === module) {
  app.listen(PORT, HOST, () => {
    console.log(`RELAY Node API Gateway listening on http://${HOST}:${PORT}`);
  });
}

module.exports = { app, createApp };

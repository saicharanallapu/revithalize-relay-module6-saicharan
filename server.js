// Express API Gateway for RELAY Module 6
// Proxies public client requests to the Python Flask microservice (port 5001)

const express = require("express");

function createApp(aiUrl = process.env.AI_URL || "http://127.0.0.1:5001") {
  const app = express();
  app.use(express.json());

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
      const response = await fetch(`${aiUrl}${internalPath}`, {
        headers: { "Accept": "application/json" },
        signal: AbortSignal.timeout(5000)
      });
      const data = await response.json();
      return res.status(response.status).json(data);
    } catch (error) {
      const isTimeout = error.name === "TimeoutError";
      return res.status(isTimeout ? 504 : 502).json({
        error: isTimeout ? "AI service timeout" : "AI service unavailable",
        message: isTimeout
          ? "Request to Flask service timed out after 5000ms."
          : `Unable to reach Python Flask microservice on ${aiUrl}`,
        details: "Ensure the Flask service is running on port 5001."
      });
    }
  }

  // Gateway health check
  app.get("/health", async (_req, res) => {
    try {
      const response = await fetch(`${aiUrl}/health`, { signal: AbortSignal.timeout(2000) });
      const aiData = await response.json();
      return res.status(200).json({
        status: "ok",
        gateway: "relay-node-gateway",
        upstreamStatus: "connected",
        aiService: aiData
      });
    } catch (err) {
      return res.status(503).json({
        status: "degraded",
        gateway: "relay-node-gateway",
        upstreamStatus: "disconnected",
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

  // Root endpoint info
  app.get("/", (_req, res) => res.json({
    name: "RELAY Module 6 API Gateway",
    version: "1.0.0",
    endpoints: [
      "GET /health",
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

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`RELAY Node API Gateway listening on http://localhost:${PORT}`);
  });
}

module.exports = { app, createApp };

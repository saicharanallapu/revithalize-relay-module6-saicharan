// Automated test suite for Node.js API Gateway (RELAY Module 6 - Two-Wheeler Focus)
// Tests:
// 1. Gateway root info endpoint
// 2. Upstream offline handling (503 / 502)
// 3. Input validation on vehicle ID (invalid characters, length)
// 4. Live Flask microservice orchestration & E2E tests:
//    - GET /health
//    - GET /api/v1/vehicles/:id/price (Motorcycles & Scooters)
//    - GET /api/v1/vehicles/:id/anomaly-score
//    - Non-existent vehicle 404 handling

const assert = require("node:assert");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { createApp } = require("./server");

function startServer(app, port, host = "127.0.0.1") {
  return new Promise((resolve, reject) => {
    const server = app.listen(port, host, () => resolve(server));
    server.on("error", reject);
  });
}

function stopServer(server) {
  return new Promise((resolve) => {
    if (!server) return resolve();
    server.close(() => resolve());
  });
}

async function waitForHealth(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (res.status === 200) return true;
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  return false;
}

async function runTests() {
  console.log("=== Running Node.js API Gateway Automated Tests (Two-Wheeler Scope) ===");
  let passed = 0;
  let failed = 0;

  function record(name, ok, err) {
    if (ok) {
      console.log(`[PASS] ${name}`);
      passed++;
    } else {
      console.error(`[FAIL] ${name}: ${err}`);
      failed++;
    }
  }

  // 1. Test Gateway with AI offline (pointing to non-existent port 59999)
  const offlineApp = createApp("http://127.0.0.1:59999");
  const offlineServer = await startServer(offlineApp, 3098);
  const offlineBase = "http://127.0.0.1:3098";

  try {
    const rHealth = await fetch(`${offlineBase}/health`);
    assert.strictEqual(rHealth.status, 503);
    const healthJson = await rHealth.json();
    assert.strictEqual(healthJson.upstreamStatus, "disconnected");
    record("Health check returns 503 when Flask upstream offline", true);

    const rPrice = await fetch(`${offlineBase}/api/v1/vehicles/RELAY_TW_001/price`);
    assert.strictEqual(rPrice.status, 502);
    const priceJson = await rPrice.json();
    assert.strictEqual(priceJson.error, "AI service unavailable");
    record("Price proxy returns 502 when Flask upstream unreachable", true);

    const rRoot = await fetch(`${offlineBase}/`);
    assert.strictEqual(rRoot.status, 200);
    const rootJson = await rRoot.json();
    assert.strictEqual(rootJson.name, "RELAY Module 6 API Gateway");
    record("Root endpoint returns API description", true);

    const rInvalidId = await fetch(`${offlineBase}/api/v1/vehicles/bad$id!/price`);
    assert.strictEqual(rInvalidId.status, 400);
    const invalidJson = await rInvalidId.json();
    assert.match(invalidJson.error, /Invalid vehicle ID/);
    record("Input validation rejects invalid vehicle ID format (400)", true);

    const r404 = await fetch(`${offlineBase}/invalid/route`);
    assert.strictEqual(r404.status, 404);
    record("Undefined routes return 404", true);
  } catch (err) {
    record("Offline gateway tests", false, err.message);
  } finally {
    await stopServer(offlineServer);
  }

  // 2. Start Flask subprocess for End-to-End Tests
  console.log("\nStarting Flask microservice for End-to-End integration tests...");
  const pythonBin = process.platform === "win32"
    ? path.join(__dirname, "ai-module", ".venv", "Scripts", "python.exe")
    : path.join(__dirname, "ai-module", ".venv", "bin", "python");

  let flaskProc = null;
  let liveServer = null;

  try {
    flaskProc = spawn(pythonBin, ["serve.py"], {
      cwd: path.join(__dirname, "ai-module"),
      stdio: "ignore",
    });

    const isLive = await waitForHealth("http://127.0.0.1:5001/health");
    if (!isLive) {
      throw new Error("Flask microservice failed to become healthy within timeout");
    }
    record("Flask microservice launched and healthy on port 5001", true);

    const liveApp = createApp("http://127.0.0.1:5001");
    liveServer = await startServer(liveApp, 3099);
    const liveBase = "http://127.0.0.1:3099";

    // E2E Health Check
    const rHealth = await fetch(`${liveBase}/health`);
    assert.strictEqual(rHealth.status, 200);
    const hData = await rHealth.json();
    assert.strictEqual(hData.status, "ok");
    assert.strictEqual(hData.upstreamStatus, "connected");
    assert.strictEqual(hData.aiService.status, "ok");
    record("E2E Gateway /health reports status:ok and upstream:connected", true);

    // E2E Price endpoint for RELAY_TW_001
    const rPrice = await fetch(`${liveBase}/api/v1/vehicles/RELAY_TW_001/price`);
    assert.strictEqual(rPrice.status, 200);
    const pData = await rPrice.json();
    assert.strictEqual(pData.vehicleId, "RELAY_TW_001");
    assert.ok(pData.fairPriceLow > 0);
    assert.ok(pData.fairPriceHigh > pData.fairPriceLow);
    assert.strictEqual(pData.currency, "INR");
    assert.ok(["Motorcycle", "Scooter", "Two-Wheeler"].includes(pData.vehicleType));
    assert.ok(pData.details);
    assert.ok(pData.details.showroomPrice > 0);
    record("E2E GET /api/v1/vehicles/RELAY_TW_001/price returns valid two-wheeler price", true);

    // E2E Price endpoint for RELAY_TW_100
    const rPrice100 = await fetch(`${liveBase}/api/v1/vehicles/RELAY_TW_100/price`);
    assert.strictEqual(rPrice100.status, 200);
    const pData100 = await rPrice100.json();
    assert.strictEqual(pData100.vehicleId, "RELAY_TW_100");
    record("E2E GET /api/v1/vehicles/RELAY_TW_100/price returns valid price", true);

    // E2E Anomaly Score for RELAY_TW_001
    const rAnomaly = await fetch(`${liveBase}/api/v1/vehicles/RELAY_TW_001/anomaly-score`);
    assert.strictEqual(rAnomaly.status, 200);
    const aData = await rAnomaly.json();
    assert.strictEqual(aData.vehicleId, "RELAY_TW_001");
    assert.ok("anomalyScore" in aData);
    assert.ok(["Suspicious", "Normal"].includes(aData.verdict));
    assert.ok(Array.isArray(aData.reasons));
    assert.ok(aData.extractedFeatures);
    record("E2E GET /api/v1/vehicles/RELAY_TW_001/anomaly-score returns score and reasons", true);

    // E2E Anomaly Score for RELAY_TW_100
    const rAnomaly100 = await fetch(`${liveBase}/api/v1/vehicles/RELAY_TW_100/anomaly-score`);
    assert.strictEqual(rAnomaly100.status, 200);
    const aData100 = await rAnomaly100.json();
    assert.strictEqual(aData100.vehicleId, "RELAY_TW_100");
    record("E2E GET /api/v1/vehicles/RELAY_TW_100/anomaly-score returns score and reasons", true);

    // E2E Price endpoint for Official Chassis ID
    const rPriceChassis = await fetch(`${liveBase}/api/v1/vehicles/ME4A185SFWPA0255V/price`);
    assert.strictEqual(rPriceChassis.status, 200);
    const pDataChassis = await rPriceChassis.json();
    assert.strictEqual(pDataChassis.vehicleId, "ME4A185SFWPA0255V");
    assert.strictEqual(pDataChassis.vehicleType, "Scooter");
    assert.strictEqual(pDataChassis.details.showroomPrice, 82000);
    assert.strictEqual(pDataChassis.details.condition, "Average");
    const baseline = pDataChassis.details.adjustedFairPrice;
    assert.strictEqual(pDataChassis.fairPriceLow, Math.round(baseline * 0.92));
    assert.strictEqual(pDataChassis.fairPriceHigh, Math.round(baseline * 1.08));
    record("E2E GET /api/v1/vehicles/ME4A185SFWPA0255V/price returns valid price for official chassis", true);

    // E2E Anomaly Score for Official Chassis ID
    const rAnomalyChassis = await fetch(`${liveBase}/api/v1/vehicles/ME4A185SFWPA0255V/anomaly-score`);
    assert.strictEqual(rAnomalyChassis.status, 200);
    const aDataChassis = await rAnomalyChassis.json();
    assert.strictEqual(aDataChassis.vehicleId, "ME4A185SFWPA0255V");
    assert.ok(["Suspicious", "Normal"].includes(aDataChassis.verdict));
    assert.ok(aDataChassis.deterministicChecks);
    assert.strictEqual(aDataChassis.deterministicChecks.rollbackWarning, true);
    record("E2E GET /api/v1/vehicles/ME4A185SFWPA0255V/anomaly-score returns evaluation and rollbackWarning for official chassis", true);

    // E2E Non-existent vehicle 404
    const rNotFound = await fetch(`${liveBase}/api/v1/vehicles/NON_EXISTENT_TW/price`);
    assert.strictEqual(rNotFound.status, 404);
    record("E2E Non-existent vehicle returns 404 through Gateway", true);

    const rNotFoundAnomaly = await fetch(`${liveBase}/api/v1/vehicles/NON_EXISTENT_TW/anomaly-score`);
    assert.strictEqual(rNotFoundAnomaly.status, 404);
    record("E2E Non-existent vehicle anomaly score returns 404 through Gateway", true);

    // E2E Admin Dashboard Web Page
    const rAdminPage = await fetch(`${liveBase}/admin`);
    assert.strictEqual(rAdminPage.status, 200);
    const adminHtml = await rAdminPage.text();
    assert.ok(adminHtml.includes("RELAY Module 6"));
    assert.ok(adminHtml.includes("ADMIN DASHBOARD"));
    assert.ok(adminHtml.includes("vehicles-table"));
    record("E2E GET /admin serves the responsive admin dashboard HTML", true);

    // E2E Admin Flagged Vehicles API
    const rFlagged = await fetch(`${liveBase}/api/v1/admin/flagged-vehicles`);
    assert.strictEqual(rFlagged.status, 200);
    const flaggedData = await rFlagged.json();
    assert.strictEqual(flaggedData.status, "ok");
    assert.strictEqual(flaggedData.totalVehicles, 500);
    assert.ok(flaggedData.returnedCount > 0);
    assert.strictEqual(flaggedData.summary.totalRollbackWarnings, 25);
    const sampleV = flaggedData.vehicles[0];
    assert.ok(sampleV.vehicleId);
    assert.ok(typeof sampleV.anomalyScore === "number");
    assert.ok(["Suspicious", "Normal"].includes(sampleV.verdict));
    assert.ok(sampleV.deterministicRollbackCheck);
    record("E2E GET /api/v1/admin/flagged-vehicles returns precomputed anomaly scores and verdicts", true);

    // E2E Admin Flagged Vehicles Filter Rollback
    const rRollbackFilter = await fetch(`${liveBase}/api/v1/admin/flagged-vehicles?filter=rollback`);
    assert.strictEqual(rRollbackFilter.status, 200);
    const rbFilterData = await rRollbackFilter.json();
    assert.strictEqual(rbFilterData.returnedCount, 25);
    record("E2E GET /api/v1/admin/flagged-vehicles?filter=rollback filters exactly 25 rollback warnings", true);

  } catch (err) {
    record("Live E2E communication test", false, err.message);
  } finally {
    await stopServer(liveServer);
    if (flaskProc) {
      flaskProc.kill();
    }
  }

  console.log(`\n========================================`);
  console.log(`Node.js Test Summary: ${passed} passed, ${failed} failed`);
  console.log(`========================================\n`);

  if (failed > 0) process.exit(1);
}

runTests();

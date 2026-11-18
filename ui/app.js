const element = (id) => document.getElementById(id);
const apiInput = element("api-url");
const tripPicker = element("trip");
const predictButton = element("predict");
const connectButton = element("connect");
const fitButton = element("fit");
const gpsToggle = element("gps-toggle");
const routeToggle = element("route-toggle");
const mapMessage = element("map-message");
const countFormat = new Intl.NumberFormat();
const percentFormat = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 1,
});

const leaflet = window.L;
const map = leaflet?.map("map", {
  preferCanvas: true,
  scrollWheelZoom: false,
  zoomAnimation: !matchMedia("(prefers-reduced-motion: reduce)").matches,
}).setView([23, 16], 2);
if (map) {
  leaflet.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map);
  leaflet.control.scale({ imperial: false }).addTo(map);
  new ResizeObserver(() => map.invalidateSize({ pan: false })).observe(element("map"));
}

const gpsLayer = map && leaflet.featureGroup().addTo(map);
const routeLayer = map && leaflet.featureGroup().addTo(map);
const colors = getComputedStyle(document.documentElement);
const gpsColor = colors.getPropertyValue("--gps").trim();
const routeColor = colors.getPropertyValue("--route").trim();
let activeUrl = "";
let tripBounds;

function status(message, error = false) {
  const feedback = element("feedback");
  feedback.textContent = message;
  feedback.dataset.error = error;
}

function showMapMessage(message) {
  mapMessage.textContent = map ? message : "Map controls could not load. Check your internet connection and reload.";
  mapMessage.hidden = !mapMessage.textContent;
}

function setConnected(connected) {
  const indicator = element("connection");
  indicator.textContent = connected ? "API connected" : "API disconnected";
  indicator.dataset.connected = connected;
}

function clearResult(message) {
  gpsLayer?.clearLayers();
  routeLayer?.clearLayers();
  tripBounds = undefined;
  fitButton.disabled = gpsToggle.disabled = routeToggle.disabled = true;
  element("gps-count").textContent = "";
  element("scores").hidden = true;
  element("scores-empty").hidden = false;
  showMapMessage(message);
}

async function request(url, options) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(45000) });
  if (!response.ok) throw new Error(`Server returned HTTP ${response.status}.`);
  return response.json();
}

function describeFailure(error) {
  if (error.name === "TimeoutError") return "The request timed out. Try again.";
  if (error instanceof TypeError) return "Check the server URL, network connection, and API CORS settings.";
  return error.message;
}

function fitTrip() {
  if (tripBounds?.isValid()) map.fitBounds(tripBounds, { padding: [30, 30], maxZoom: 17, animate: false });
}

function coordinates(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.filter((pair) => Array.isArray(pair) && pair.length === 2 && pair.every(Number.isFinite));
}

function syncLayer(layer, toggle) {
  if (!map) return;
  if (toggle.checked) {
    layer.addTo(map);
    if (layer === routeLayer && gpsToggle.checked) gpsLayer.bringToFront();
  } else {
    layer.removeFrom(map);
  }
}

function drawTrip(result) {
  const gps = coordinates(result.gps);
  const route = coordinates(result.pred_route_polyline);
  const stride = Math.max(1, Math.ceil(gps.length / 3000));

  if (map) {
    for (let i = 0; i < gps.length; i += stride) {
      leaflet.circleMarker(gps[i], {
        radius: 3, stroke: false, fillColor: gpsColor, fillOpacity: 0.75, interactive: false,
      }).addTo(gpsLayer);
    }
    if (route.length > 1) {
      leaflet.polyline(route, { color: "#fff", weight: 8, opacity: 0.9, interactive: false }).addTo(routeLayer);
      leaflet.polyline(route, { color: routeColor, weight: 4, opacity: 0.95, interactive: false }).addTo(routeLayer);
      // Draw dots above the route so coincident GPS observations remain visible.
      if (gpsToggle.checked) gpsLayer.bringToFront();
    }

    tripBounds = leaflet.latLngBounds(gps.concat(route.length > 1 ? route : []));
    fitButton.disabled = !tripBounds.isValid();
    gpsToggle.disabled = !gps.length;
    routeToggle.disabled = route.length < 2;
    fitTrip();
  }

  const count = result.n_obs ?? gps.length;
  const shown = Math.ceil(gps.length / stride);
  element("gps-count").textContent = `${countFormat.format(count)} GPS points${stride > 1 ? ` (${countFormat.format(shown)} plotted)` : ""}`;
  showMapMessage(route.length > 1 ? "" : "The server returned no matched route for this trip.");
}

function displayScores(result) {
  const metrics = result.metrics;
  const pct = (x) => Number.isFinite(x) ? percentFormat.format(x) : "Unavailable";
  const count = (x) => Number.isFinite(x) ? countFormat.format(x) : "Unavailable";
  const values = {
    "top1-clf": pct(metrics?.classifier_only?.top1_accuracy),
    "top1-hmm": pct(metrics?.hmm?.top1_accuracy),
    "f1-clf": pct(metrics?.classifier_only?.route_arc_f1),
    "f1-hmm": pct(metrics?.hmm?.route_arc_f1),
    "cont-clf": pct(result.continuity_clf),
    "cont-hmm": pct(result.continuity_hmm),
    scored: count(metrics?.n_valid_points),
    breaks: count(result.route_breaks),
    elapsed: Number.isFinite(result.timing_s?.total) ? `${result.timing_s.total.toFixed(2)} s` : "Unavailable",
  };
  for (const [id, value] of Object.entries(values)) element(id).textContent = value;
  element("scores").hidden = false;
  element("scores-empty").hidden = true;
}

element("connection-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = apiInput.value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(url)) {
    status("Enter an HTTP or HTTPS server URL.", true);
    return;
  }

  activeUrl = "";
  setConnected(false);
  connectButton.disabled = apiInput.disabled = tripPicker.disabled = predictButton.disabled = true;
  tripPicker.replaceChildren(new Option("Loading trips", ""));
  clearResult("Loading trips from the prediction server.");
  status("Connecting to the prediction server…");

  try {
    const payload = await request(`${url}/trips`);
    const trips = Array.isArray(payload) ? payload : payload?.trips;
    if (!Array.isArray(trips)) throw new Error("The server returned an invalid trip list.");
    const ids = trips.map((trip) => trip?.trip_id ?? trip?.id ?? trip)
      .filter((id) => typeof id === "string" || Number.isFinite(id));
    activeUrl = url;
    setConnected(true);
    tripPicker.replaceChildren(...(ids.length ? ids.map((id) => new Option(String(id), String(id))) : [new Option("No trips available", "")]));
    tripPicker.disabled = predictButton.disabled = !ids.length;
    status(ids.length ? `${ids.length} ${ids.length === 1 ? "trip" : "trips"} available.` : "No trips found. Check the server’s trip folder.");
    showMapMessage(ids.length ? "Select a trip and match its route." : "The prediction server has no trips to display.");
  } catch (error) {
    tripPicker.replaceChildren(new Option("Connect to load trips", ""));
    status(`Could not load trips. ${describeFailure(error)}`, true);
    showMapMessage("Connect to a prediction server to load a trip.");
  } finally {
    connectButton.disabled = apiInput.disabled = false;
  }
});

element("prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!activeUrl || !tripPicker.value) return;
  const tripId = tripPicker.value;
  connectButton.disabled = apiInput.disabled = tripPicker.disabled = predictButton.disabled = true;
  clearResult("Matching the GPS observations to roads…");
  status(`Matching trip ${tripId}…`);

  try {
    const result = await request(`${activeUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    if (!Array.isArray(result?.gps) || !Array.isArray(result.pred_route_polyline)) {
      throw new Error("The server returned an invalid prediction.");
    }
    drawTrip(result);
    displayScores(result);
    status(`Prediction received for trip ${tripId}.`);
  } catch (error) {
    status(`Could not match trip ${tripId}. ${describeFailure(error)}`, true);
    showMapMessage("Select a trip and try matching again.");
  } finally {
    connectButton.disabled = apiInput.disabled = tripPicker.disabled = predictButton.disabled = false;
  }
});

tripPicker.addEventListener("change", () => {
  clearResult("Match the selected trip to view its route.");
  status("");
});
apiInput.addEventListener("input", () => {
  activeUrl = "";
  setConnected(false);
  tripPicker.disabled = predictButton.disabled = true;
  tripPicker.replaceChildren(new Option("Connect to load trips", ""));
  clearResult("Connect to a prediction server to load a trip.");
  status("");
});
fitButton.addEventListener("click", fitTrip);
gpsToggle.addEventListener("change", () => syncLayer(gpsLayer, gpsToggle));
routeToggle.addEventListener("change", () => syncLayer(routeLayer, routeToggle));
showMapMessage(mapMessage.textContent);

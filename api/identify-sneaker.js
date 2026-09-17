// Server-side sneaker identification endpoint.
//
// This only runs when the catalog is deployed somewhere with real serverless
// functions (e.g. this repo imported into Vercel) and viewed OUTSIDE the
// Claude artifact runtime. Inside the artifact, identification instead goes
// through the `sample` capability (see index.html) — the viewer's own Claude
// account answers directly, with no API key anywhere near the browser.
//
// Required environment variables (set in the Vercel project settings, never
// committed to the repo):
//   ANTHROPIC_API_KEY  - secret key from console.anthropic.com
//   ANTHROPIC_MODEL    - optional, defaults to a current vision-capable model
//
// POST body (application/json):
//   { productId, sourceFilename, imageBase64, mediaType, existingName }
// Response (application/json), always this shape on success:
//   { name, brand, modelFamily, colorway, visibleProductCode,
//     confidence, verificationStatus, alternativeCandidates, needsHumanReview }

const Anthropic = require('@anthropic-ai/sdk');

const DEFAULT_MODEL = 'claude-sonnet-5';
const ALLOWED_MEDIA_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
const MAX_BASE64_BYTES = 20 * 1024 * 1024; // ~20MB decoded, matches Anthropic's own image cap
const REQUEST_TIMEOUT_MS = 55000;

function buildPrompt(existingName) {
  return (
    'Eres un experto en identificar modelos de zapatillas (sneakers) por su apariencia visual. ' +
    'Analiza la foto considerando: marcas visibles, silueta, estructura de la suela, paneles y materiales, ' +
    'construcción de lengüeta y talón, texto visible, etiquetas de caja, códigos de producto (SKU) y ' +
    'combinación de colores / colorways oficiales conocidos. ' +
    (existingName ? ('El catálogo ya tiene un nombre provisional para este producto: "' + existingName + '". Verifícalo o corrígelo si la foto muestra algo distinto. ') : '') +
    'Responde SOLO con un objeto JSON con EXACTAMENTE estas claves: ' +
    'name (string, formato "Marca + Modelo + Edición/Silueta + Colorway", ej: "Nike Air Jordan 4 Retro Military Black" — ' +
    'solo un ejemplo de formato, nunca lo asignes salvo que la foto lo respalde), ' +
    'brand (string o null), modelFamily (string o null), colorway (string o null), ' +
    'visibleProductCode (string o null, solo si se lee un código/SKU en la foto), ' +
    'confidence (número entre 0 y 1, prudente — nunca 1.0 solo por parecido visual), ' +
    'verificationStatus ("verified" si un código de producto legible confirma el modelo, "probable" si la seña visual es fuerte pero sin código legible, "uncertain" si hay dudas relevantes), ' +
    'alternativeCandidates (array de hasta 3 strings con otros nombres posibles, o array vacío), ' +
    'needsHumanReview (boolean, true si confidence < 0.85). ' +
    'Da siempre tu mejor nombre posible en "name" aunque la confianza sea baja — NUNCA dejes "name" vacío ni uses un texto genérico tipo "Modelo por confirmar"; la incertidumbre se expresa con confidence/verificationStatus, no con el nombre. ' +
    'No agregues texto fuera del objeto JSON.'
  );
}

function clampConfidence(value) {
  const n = Number(value);
  if (!isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

// Validates and coerces the model's reply into the exact response contract.
// Never trusts the model's JSON blindly — every field is checked and clamped.
function validateIdentification(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new Error('invalid_model_output');
  }
  const verificationStatus = ['verified', 'probable', 'uncertain'].includes(raw.verificationStatus)
    ? raw.verificationStatus
    : 'uncertain';
  const alternativeCandidates = Array.isArray(raw.alternativeCandidates)
    ? raw.alternativeCandidates.map((s) => String(s || '').trim()).filter(Boolean).slice(0, 3)
    : [];
  const confidence = clampConfidence(raw.confidence);
  return {
    name: String(raw.name || '').trim(),
    brand: raw.brand ? String(raw.brand).trim() : null,
    modelFamily: raw.modelFamily ? String(raw.modelFamily).trim() : null,
    colorway: raw.colorway ? String(raw.colorway).trim() : null,
    visibleProductCode: raw.visibleProductCode ? String(raw.visibleProductCode).trim() : null,
    confidence,
    verificationStatus,
    alternativeCandidates,
    needsHumanReview: Boolean(raw.needsHumanReview) || confidence < 0.85,
  };
}

// Extracts the first parseable JSON object from the model's text reply.
// Claude is instructed to reply with only JSON, but this tolerates a stray
// code fence or a sentence around it rather than failing outright.
function extractJsonObject(text) {
  const direct = tryParse(text);
  if (direct) return direct;
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced) {
    const parsed = tryParse(fenced[1]);
    if (parsed) return parsed;
  }
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start !== -1 && end !== -1 && end > start) {
    const parsed = tryParse(text.slice(start, end + 1));
    if (parsed) return parsed;
  }
  return null;
}
function tryParse(s) {
  try { return JSON.parse(s); } catch (e) { return null; }
}

async function readJsonBody(req) {
  if (req.body && typeof req.body === 'object') return req.body; // Vercel Node runtime auto-parses JSON
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => { raw += chunk; });
    req.on('end', () => {
      try { resolve(raw ? JSON.parse(raw) : {}); } catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    // Never expose the missing-key detail beyond "not configured" — and
    // never, ever, embed a fallback key in code.
    res.status(500).json({ error: 'server_not_configured', message: 'ANTHROPIC_API_KEY no está configurada en el servidor.' });
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (e) {
    res.status(400).json({ error: 'invalid_request', message: 'Cuerpo de la petición inválido.' });
    return;
  }

  const { productId, sourceFilename, imageBase64, mediaType, existingName } = body || {};

  if (!imageBase64 || typeof imageBase64 !== 'string') {
    res.status(400).json({ error: 'invalid_image', message: 'Falta la imagen o el archivo está dañado.' });
    return;
  }
  if (!ALLOWED_MEDIA_TYPES.includes(mediaType)) {
    res.status(400).json({ error: 'unsupported_format', message: 'Formato no soportado. Usa JPG, PNG, WebP o GIF.' });
    return;
  }
  const approxBytes = Math.ceil((imageBase64.length * 3) / 4);
  if (approxBytes > MAX_BASE64_BYTES) {
    res.status(400).json({ error: 'image_too_large', message: 'La imagen es demasiado grande (máx. 20MB).' });
    return;
  }

  const client = new Anthropic({ apiKey });
  const model = process.env.ANTHROPIC_MODEL || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  // Limited retries with exponential backoff for transient failures only —
  // never retried on a bad request or an auth error, and never more than 3 tries total.
  const maxAttempts = 3;
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const message = await client.messages.create({
        model,
        max_tokens: 500,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: mediaType, data: imageBase64 } },
            { type: 'text', text: buildPrompt(existingName) },
          ],
        }],
      }, { signal: controller.signal });

      clearTimeout(timeout);
      const textBlock = (message.content || []).find((b) => b.type === 'text');
      const rawText = textBlock ? textBlock.text : '';
      if (!rawText.trim()) throw Object.assign(new Error('empty_completion'), { retryable: false });

      const parsed = extractJsonObject(rawText);
      if (!parsed) throw Object.assign(new Error('invalid_json'), { retryable: false });

      const identification = validateIdentification(parsed);
      res.status(200).json(Object.assign({ productId: productId || null, sourceFilename: sourceFilename || null }, identification));
      return;
    } catch (err) {
      lastError = err;
      const status = err && err.status;
      const retryable = status === 429 || status === 500 || status === 502 || status === 503 || status === 529 || err.name === 'AbortError';
      if (!retryable || attempt === maxAttempts) break;
      await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt - 1)));
    }
  }

  clearTimeout(timeout);
  const status = lastError && lastError.status;
  if (status === 401 || status === 403) {
    res.status(500).json({ error: 'unauthorized', message: 'La API key de Anthropic fue rechazada. Revisa ANTHROPIC_API_KEY en Vercel.' });
  } else if (status === 429) {
    res.status(429).json({ error: 'rate_limited', message: 'Límite de solicitudes alcanzado. Intenta de nuevo en unos segundos.' });
  } else if (lastError && lastError.name === 'AbortError') {
    res.status(504).json({ error: 'timeout', message: 'La identificación tardó demasiado. Intenta de nuevo.' });
  } else if (lastError && (lastError.message === 'invalid_json' || lastError.message === 'empty_completion')) {
    res.status(502).json({ error: 'invalid_model_output', message: 'La IA no devolvió una respuesta utilizable.' });
  } else {
    res.status(502).json({ error: 'upstream_error', message: 'No se pudo completar la identificación. Intenta de nuevo.' });
  }
};

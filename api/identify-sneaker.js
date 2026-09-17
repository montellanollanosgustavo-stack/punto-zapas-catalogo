// Server-side sneaker identification endpoint.
//
// This only runs when the catalog is deployed somewhere with real serverless
// functions (e.g. this repo imported into Vercel) and viewed OUTSIDE the
// Claude artifact runtime. Inside the artifact, identification instead goes
// through the `sample` capability (see index.html) — the viewer's own Claude
// account answers directly, with no API key anywhere near the browser.
//
// Required environment variables (set in the Vercel project settings — see
// .env.example — never committed to the repo):
//   ANTHROPIC_API_KEY  - secret key from console.anthropic.com
//   ANTHROPIC_MODEL    - optional, defaults to a current vision-capable model
//
// Request: multipart/form-data (NOT JSON, NOT a blob: URL) —
//   image          - the ORIGINAL File the browser read from disk
//   productId      - the product's own stable id, echoed back unchanged
//   existingName   - optional, a provisional name already on the product
//
// Response (application/json), always this shape on success:
//   { productId, exactName, brand, model, colorway, confidence,
//     alternatives, needsReview,
//     -- extra fields the frontend also understands --
//     visibleProductCode, verificationStatus }

const Busboy = require('busboy');
const Anthropic = require('@anthropic-ai/sdk');

const DEFAULT_MODEL = 'claude-sonnet-5';
const ALLOWED_MEDIA_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
const MAX_IMAGE_BYTES = 20 * 1024 * 1024; // matches Anthropic's own per-image cap
const REQUEST_TIMEOUT_MS = 25000; // per attempt — 3 attempts worst case stays well under typical serverless limits

function buildPrompt(existingName) {
  return (
    'Analiza esta imagen de una zapatilla (sneaker) e identifica el producto comercial real ' +
    'con la mayor precisión que la evidencia visible permita: silueta visible, paneles, suela, ' +
    'logos, materiales, combinación de colores, etiquetas y códigos de producto. ' +
    (existingName ? ('El catálogo ya tiene un nombre provisional: "' + existingName + '". Verifícalo o corrígelo si la foto muestra algo distinto. ') : '') +
    'Responde SOLO con un objeto JSON con EXACTAMENTE estas claves: ' +
    'brand (string o null), model (string o null), colorway (string o null), ' +
    'exactName (string — el nombre comercial real más probable, ej: "Nike Air Jordan 4 Retro Military Black"), ' +
    'confidence (número de 0 a 1), alternatives (array de hasta 3 strings), needsReview (boolean), ' +
    'visibleProductCode (string o null, solo si se lee un código/SKU en la foto), ' +
    'verificationStatus ("verified" si un código de producto legible confirma el modelo, "probable" si la seña visual es fuerte pero sin código legible, "uncertain" si hay dudas relevantes). ' +
    'No inventes un SKU, colaboración, edición o colorway que la imagen no respalde. ' +
    'exactName debe usar el nombre comercial real más probable. ' +
    'NUNCA devuelvas "Modelo por confirmar", "Desconocido", "Sneaker" u otro texto genérico como exactName. ' +
    'Si la certeza es limitada, da el nombre mejor respaldado, pon needsReview en true e incluye alternatives. ' +
    'No agregues texto fuera del objeto JSON.'
  );
}

function clampConfidence(value) {
  const n = Number(value);
  if (!isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

const GENERIC_NAMES = ['modelo por confirmar', 'desconocido', 'sneaker', 'unknown', 'n/a', ''];

// Validates and coerces the model's reply into the exact response contract.
// Never trusts the model's JSON blindly — every field is checked and clamped.
function validateIdentification(raw) {
  if (!raw || typeof raw !== 'object') {
    throw Object.assign(new Error('invalid_model_output'), { retryable: false });
  }
  const verificationStatus = ['verified', 'probable', 'uncertain'].includes(raw.verificationStatus)
    ? raw.verificationStatus
    : 'uncertain';
  const alternatives = Array.isArray(raw.alternatives)
    ? raw.alternatives.map((s) => String(s || '').trim()).filter(Boolean).slice(0, 3)
    : [];
  const confidence = clampConfidence(raw.confidence);
  let exactName = String(raw.exactName || '').trim();
  if (GENERIC_NAMES.includes(exactName.toLowerCase())) exactName = ''; // never forward a placeholder as if it were real
  return {
    exactName,
    brand: raw.brand ? String(raw.brand).trim() : null,
    model: raw.model ? String(raw.model).trim() : null,
    colorway: raw.colorway ? String(raw.colorway).trim() : null,
    visibleProductCode: raw.visibleProductCode ? String(raw.visibleProductCode).trim() : null,
    confidence,
    verificationStatus,
    alternatives,
    needsReview: Boolean(raw.needsReview) || confidence < 0.85 || !exactName,
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

// Parses the multipart/form-data body: the ORIGINAL file bytes plus the plain
// text fields. Rejects anything over MAX_IMAGE_BYTES while streaming, rather
// than buffering an unbounded upload first.
function parseMultipart(req) {
  return new Promise((resolve, reject) => {
    let busboy;
    try {
      busboy = Busboy({ headers: req.headers, limits: { fileSize: MAX_IMAGE_BYTES, files: 1 } });
    } catch (e) {
      reject(Object.assign(new Error('invalid_request'), { code: 'invalid_request' }));
      return;
    }
    const fields = {};
    let file = null;
    let fileTooLarge = false;

    busboy.on('field', (name, value) => { fields[name] = value; });
    busboy.on('file', (name, stream, info) => {
      if (name !== 'image') { stream.resume(); return; }
      const chunks = [];
      stream.on('data', (chunk) => chunks.push(chunk));
      stream.on('limit', () => { fileTooLarge = true; });
      stream.on('end', () => {
        file = { buffer: Buffer.concat(chunks), mimeType: info.mimeType, filename: info.filename };
      });
    });
    busboy.on('error', (err) => reject(err));
    busboy.on('finish', () => {
      if (fileTooLarge) { reject(Object.assign(new Error('image_too_large'), { code: 'image_too_large' })); return; }
      resolve({ fields, file });
    });
    req.pipe(busboy);
  });
}

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    // Never expose more than "not configured" — and never, ever, embed a fallback key.
    res.status(500).json({ error: 'server_not_configured', message: 'Falta configurar ANTHROPIC_API_KEY en el servidor.' });
    return;
  }

  let parsed;
  try {
    parsed = await parseMultipart(req);
  } catch (e) {
    if (e && e.code === 'image_too_large') {
      res.status(400).json({ error: 'image_too_large', message: 'La imagen es demasiado grande (máx. 20MB).' });
    } else {
      res.status(400).json({ error: 'invalid_request', message: 'No se pudo leer la imagen enviada.' });
    }
    return;
  }

  const { fields, file } = parsed;
  const productId = fields.productId || null;
  const existingName = fields.existingName || null;

  if (!file || !file.buffer || !file.buffer.length) {
    res.status(400).json({ error: 'invalid_image', message: 'Falta la imagen o el archivo está dañado.', productId });
    return;
  }
  if (!ALLOWED_MEDIA_TYPES.includes(file.mimeType)) {
    res.status(400).json({ error: 'unsupported_format', message: 'Formato no soportado. Usa JPG, PNG, WebP o GIF.', productId });
    return;
  }

  const imageBase64 = file.buffer.toString('base64');
  const client = new Anthropic({ apiKey });
  const model = process.env.ANTHROPIC_MODEL || DEFAULT_MODEL;

  // Limited retries with exponential backoff for transient failures only.
  // Each attempt gets its OWN AbortController: reusing one across retries
  // would leave it permanently aborted after the first timeout, making
  // every later attempt fail instantly instead of actually retrying.
  const maxAttempts = 3;
  let lastError = null;
  let lastAborted = false;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const message = await client.messages.create({
        model,
        max_tokens: 500,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: file.mimeType, data: imageBase64 } },
            { type: 'text', text: buildPrompt(existingName) },
          ],
        }],
      }, { signal: controller.signal });

      clearTimeout(timeout);
      const textBlock = (message.content || []).find((b) => b.type === 'text');
      const rawText = textBlock ? textBlock.text : '';
      if (!rawText.trim()) throw Object.assign(new Error('empty_completion'), { retryable: false });

      const parsedJson = extractJsonObject(rawText);
      if (!parsedJson) throw Object.assign(new Error('invalid_json'), { retryable: false });

      const identification = validateIdentification(parsedJson);
      res.status(200).json(Object.assign({ productId }, identification));
      return;
    } catch (err) {
      clearTimeout(timeout);
      lastError = err;
      lastAborted = controller.signal.aborted;
      const status = err && err.status;
      const explicitlyNonRetryable = err && err.retryable === false;
      const retryable = !explicitlyNonRetryable && (
        lastAborted || status === 429 || status === 500 || status === 502 || status === 503 || status === 529 || status === undefined
      );
      if (!retryable || attempt === maxAttempts) break;
      await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt - 1)));
    }
  }

  const status = lastError && lastError.status;
  if (status === 401 || status === 403) {
    res.status(500).json({ error: 'unauthorized', message: 'La API key de Anthropic fue rechazada. Revisa ANTHROPIC_API_KEY en Vercel.', productId });
  } else if (status === 429) {
    res.status(429).json({ error: 'rate_limited', message: 'Límite de solicitudes alcanzado. Intenta de nuevo en unos segundos.', productId });
  } else if (lastAborted) {
    res.status(504).json({ error: 'timeout', message: 'La identificación tardó demasiado. Intenta de nuevo.', productId });
  } else if (lastError && (lastError.message === 'invalid_json' || lastError.message === 'empty_completion')) {
    res.status(502).json({ error: 'invalid_model_output', message: 'La IA no devolvió una respuesta utilizable.', productId });
  } else {
    res.status(502).json({ error: 'upstream_error', message: 'No se pudo completar la identificación. Intenta de nuevo.', productId });
  }
};

# syntax=docker/dockerfile:1

# Container-Image der Demo. Modell und Mandantenindizes werden im Build
# eingebacken; zur Laufzeit entsteht kein Index und wird nichts nachgeladen.
#
# Kein HEALTHCHECK. Azure Container Apps prueft ueber Startup-, Liveness- und
# Readiness-Proben auf der Container-App-Ressource, nicht ueber die
# Docker-Anweisung; ein HEALTHCHECK hier waere wirkungslos und wuerde einen
# Mechanismus vortaeuschen, den die Plattform nicht liest.
# Quelle: https://learn.microsoft.com/azure/container-apps/health-probes

# ============================================================================
# Stage 1 -- Builder
# ============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Eigenes venv, damit in den Runtime-Stage nur die Abhaengigkeiten wandern und
# nicht das Werkzeug, mit dem sie installiert wurden.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Die Anwendung liegt als QUELLE unter /app und wird editierbar installiert.
# Das ist keine Bequemlichkeit, sondern notwendig:
#
#   app/config.py leitet die Projektwurzel aus dem Ort DIESER Datei ab, und
#   app/main.py leitet das Verzeichnis der Oberflaeche genauso ab. Waere das
#   Paket nach site-packages kopiert, zeigten beide dorthin: Die Mandanten
#   waeren nicht auffindbar und die Oberflaeche nicht ausgeliefert -- ohne
#   Fehlermeldung, weil beide Stellen ein fehlendes Verzeichnis still
#   hinnehmen. Fuer die Mandanten gibt es eine Umgebungsvariable, mit der man
#   das geradebiegen koennte; fuer die Oberflaeche gibt es keine.
#
#   Editierbar installiert bleiben app/ und static/ Geschwister unter /app,
#   und beide Ableitungen stimmen. Es gibt genau eine Kopie des Codes im Image.
#
#   FOLGE, DIE MAN NICHT AUFRAEUMEN DARF: Builder- und Laufzeitstufe muessen
#   denselben ABSOLUTEN Pfad benutzen. Der editierbare Einbau legt in
#   site-packages eine Datei ab, die den Quellpfad woertlich enthaelt --
#   hier eine .pth mit genau dem Inhalt "/app". Wandert der Code in der
#   Laufzeitstufe an eine andere Stelle, zeigt dieser Eintrag ins Leere und
#   der Import des Pakets scheitert. Ein WORKDIR /srv oder ein
#   COPY --from=builder /app /opt/app in der zweiten Stufe reicht dafuer aus.
#
#   Wer hier aufraeumen will, prueft zuerst:
#     docker run --rm --entrypoint sh IMAGE -c \
#       'cat /opt/venv/lib/python3.12/site-packages/*_editable_*.pth'
COPY pyproject.toml ./
COPY app ./app
COPY static ./static
COPY scripts ./scripts
RUN pip install --editable .

# --- Embedding-Modell einbacken --------------------------------------------
# HF_HOME legt fest, wohin die Bibliothek ihre Daten schreibt; der Modellcache
# liegt darunter. Die Variablen werden beim Import gelesen, stehen also vor
# jedem Aufruf fest.
# Quelle: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
ENV HF_HOME=/opt/hf

# Der Modellname wird aus der Konfiguration GELESEN, nicht hier wiederholt.
# Ein zweiter Ort fuer denselben Wert waere die Doppeldeutigkeit, die spaeter
# Zeit kostet -- und ein Modellwechsel wuerde hier stillschweigend danebenlaufen.
# Gelesen wird das Klassenfeld, ohne die Konfiguration zu bauen: Letzteres
# verlangte einen Schluessel, den es hier nicht gibt.
RUN python -c "\
from app.config import Settings; \
from sentence_transformers import SentenceTransformer; \
name = Settings.model_fields['embedding_model'].default; \
print('lade', name); \
SentenceTransformer(name)"

# --- Build-Gate -------------------------------------------------------------
# Der volle Kontext liegt unter /kontext und dient AUSSCHLIESSLICH dazu, die
# Allowlist zu berechnen. Er wandert nicht in den Runtime-Stage.
COPY tenants /kontext/tenants

RUN python scripts/pruefe_public_image.py \
        --allowlist-schreiben /app/erlaubte-mandanten.txt \
        --tenants-dir /kontext/tenants

# Nur die berechnete Liste wird kopiert. Kein COPY tenants/ am Stueck.
RUN mkdir -p /app/tenants && \
    while IFS= read -r slug; do \
        cp -r "/kontext/tenants/$slug" "/app/tenants/$slug"; \
    done < /app/erlaubte-mandanten.txt

# Und danach die Gegenprobe: Mengengleichheit in beide Richtungen. Bricht ab,
# wenn ein Mandant zuviel kopiert wurde ODER einer fehlt.
RUN python scripts/pruefe_public_image.py \
        --pruefen \
        --tenants-dir /app/tenants \
        --allowlist /app/erlaubte-mandanten.txt

# --- Indizes bauen ----------------------------------------------------------
# Der Ingest baut die Konfiguration ueber pydantic-settings auf, und die
# verlangt den Schluessel des gewaehlten Anbieters -- obwohl der Indexbau kein
# Sprachmodell benutzt.
#
# Der Platzhalter steht deshalb NUR vor diesem einen Kommando: kein ENV, das
# ins Image wanderte, und auch kein ARG. Ein ARG waere zwar ebenfalls nicht
# eingebettet, liesse sich aber mit --build-arg ueberschreiben -- und ein
# echter Schluessel hat im Bau nichts zu suchen. So ist das ausgeschlossen
# statt nur unerwuenscht. Nebenwirkung: Der Bau-Linter warnt nicht mehr ueber
# einen schluesselfoermigen Namen in ARG oder ENV, und die Warnung bleibt
# aussagekraeftig fuer den Fall, in dem sie wirklich etwas bedeutet.
#
# Kein TENANTS_DIR, kein INDEX_DIR: Die Voreinstellungen der Konfiguration
# loesen gegen /app auf und treffen damit bereits das Richtige. Denselben Wert
# ein zweites Mal zu setzen, macht ihn zweideutig statt sicherer -- und die
# Gegenprobe oben weist nach, dass er stimmt.
RUN OPENAI_API_KEY=indexbau-ohne-llm python -m app.ingest --all

# ============================================================================
# Stage 2 -- Laufzeit
# ============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    UVICORN_PORT=8000

# Kein Netzzugriff beim Start: Ohne diese Variable prueft die Bibliothek auch
# bei vollstaendigem Cache per HTTP, ob eine neuere Fassung vorliegt. Im
# Container waere das ein Aufruf nach draussen auf dem Kaltstartpfad -- und bei
# gesperrtem Netz eine Verzoegerung, deren Ursache niemand sieht.
# Quelle: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
ENV HF_HUB_OFFLINE=1

# Nicht --system: Das reserviert UIDs unter 1000, und eine hohe UID gaebe bei
# jedem Bau eine Warnung, die nichts bedeutet. Die UID ist trotzdem fest,
# damit Dateirechte ueber Neubauten hinweg gleich bleiben.
RUN useradd --create-home --uid 10001 demo

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
COPY --from=builder --chown=demo:demo /app /app

WORKDIR /app
USER demo

# Nur Dokumentation -- der tatsaechliche Port kommt aus UVICORN_PORT.
EXPOSE 8000

# Exec-Form, damit das Signal des Orchestrators den Prozess erreicht und nicht
# in einer Shell haengenbleibt. Der Port steht deshalb nicht im Kommando: Die
# uvicorn-Kommandozeile liest jede Option auch aus UVICORN_<OPTION>, und damit
# bleibt er einstellbar, ohne dass eine Shell die Variable ersetzen muss.
# Die Bindung auf 0.0.0.0 steht dagegen fest im Kommando -- sie ist eine
# Sicherheitsaussage und soll nicht beilaeufig ueberschreibbar sein.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0"]

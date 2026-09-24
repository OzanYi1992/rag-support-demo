"""Tests des Providerzugangs - und zwar am PAYLOAD, nicht am Objekt.

Der Grund fuer diese Datei ist ein Fehlschlag vom 2026-09-24. Die erste Fassung
der Temperaturdurchreichung wurde gegen ein Settings-Objekt mit dem
Modellnamen "x" geprueft. Der Wert stand danach brav am Modellobjekt, der Test
war gruen - und im Betrieb fehlte der Parameter im Payload, weil
langchain-openai ihn fuer die gpt-5-Familie still verwirft. Ein gruener Test,
dessen Gegenstand der falsche war (P-023).

Deshalb prueft hier alles den Payload, den `_get_request_payload` liefert, und
mit den ECHTEN Modellnamen. Netz wird dabei nicht angefasst: Der Payload
entsteht vor dem Absenden.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models.base import BaseChatOpenAI

from app.config import Settings
from app.llm import build_llm
from app.prompts import antwortmodell_fuer

# Das Modell, das im Betrieb konfiguriert ist. Steht hier als Literal, damit
# der Test nicht mitwandert, wenn jemand die Konfiguration aendert - dann soll
# er auffallen.
MODELL_IM_BETRIEB = "gpt-5.4-mini-2026-03-17"

# Ein Modell aus einer Familie, die temperature annimmt. Die Gegenprobe zur
# Aussage "der Parameter wird verworfen" - ohne sie waere nicht zu
# unterscheiden, ob unser Code ihn nie setzt.
MODELL_MIT_TEMPERATUR = "gpt-4o-mini"

NACHRICHTEN = [SystemMessage(content="system"), HumanMessage(content="user")]


def _payload(modellname: str, temperatur: float) -> dict:
    """Der Payload, den der fertig gebaute Client absenden wuerde."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key="platzhalter",
        openai_model=modellname,
        llm_temperature=temperatur,
    )
    klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
    gebunden = _bindung(klient._runnable)
    assert gebunden is not None, "Chatmodell in der Kette nicht gefunden"
    return gebunden.bound._get_request_payload(NACHRICHTEN, stop=None, **gebunden.kwargs)


def _bindung(objekt: object, tiefe: int = 0) -> object | None:
    """Sucht das gebundene Chatmodell in der Kette.

    Rekursiv, weil `include_raw=True` die Kette verzweigt: Das Modell liegt
    unter steps[0].steps__["raw"] und nicht auf der obersten Ebene.
    """
    if tiefe > 6:
        return None
    if isinstance(getattr(objekt, "bound", None), BaseChatOpenAI):
        return objekt
    for name in ("steps", "middle"):
        if isinstance(getattr(objekt, name, None), list):
            for teil in getattr(objekt, name):
                gefunden = _bindung(teil, tiefe + 1)
                if gefunden is not None:
                    return gefunden
    abbildung = getattr(objekt, "steps__", None)
    if isinstance(abbildung, dict):
        for teil in abbildung.values():
            gefunden = _bindung(teil, tiefe + 1)
            if gefunden is not None:
                return gefunden
    for name in ("first", "last", "bound", "runnable", "default"):
        teil = getattr(objekt, name, None)
        if teil is not None and teil is not objekt:
            gefunden = _bindung(teil, tiefe + 1)
            if gefunden is not None:
                return gefunden
    return None


def test_temperatur_erreicht_den_payload():
    """Der konfigurierte Wert wird gesendet, nicht nur gesetzt.

    Das ist der Zweck der Einstellung: Ohne sie entschied der Anbieter, und eine
    Aenderung seines Standards haette jede Messung verschoben, ohne dass eine
    Zeile Code anders wird.
    """
    assert _payload(MODELL_MIT_TEMPERATUR, 0.0)["temperature"] == 0.0
    assert _payload(MODELL_MIT_TEMPERATUR, 0.7)["temperature"] == 0.7


def test_nur_temperatur_eins_erreicht_das_konfigurierte_modell():
    """Die gpt-5-Familie nimmt ausschliesslich temperature=1.

    langchain-openai verwirft jeden anderen Wert in einem Validator, STILL -
    kein Fehler, keine Warnung, der Parameter fehlt danach im Payload.

    Dieser Test haelt die Einschraenkung fest, damit niemand den Standard auf
    0.0 dreht und glaubt, er wirke. Wird er eines Tages rot, weil auch 0.0
    ankommt, dann hat der Anbieter oder die Bibliothek die Einschraenkung
    aufgehoben - und dann ist die Wahl der Temperatur neu zu messen, nicht der
    Test zu loeschen.
    """
    assert _payload(MODELL_IM_BETRIEB, 1.0)["temperature"] == 1.0
    assert "temperature" not in _payload(MODELL_IM_BETRIEB, 0.0)


def test_das_strukturschema_folgt_der_uebergebenen_sprache():
    """Was beim Modell ankommt, ist das Schema der Mandantensprache.

    Geprueft am Payload und nicht am Klassenobjekt: Zwischen beiden liegt
    `with_structured_output`, und genau dort ist der Fehler vom 2026-09-24
    unentdeckt durchgelaufen.
    """
    from openai.lib._parsing._completions import type_to_response_format_param

    for sprache, erwartet in (("de", "Die Antwort"), ("en", "The answer")):
        settings = Settings(
            llm_provider="openai",
            openai_api_key="platzhalter",
            openai_model=MODELL_IM_BETRIEB,
            llm_temperature=1.0,
        )
        klient = build_llm(settings, None, answer_model=antwortmodell_fuer(sprache))
        gebunden = _bindung(klient._runnable)
        payload = gebunden.bound._get_request_payload(NACHRICHTEN, stop=None, **gebunden.kwargs)
        schema = type_to_response_format_param(payload["response_format"])
        beschreibung = schema["json_schema"]["schema"]["properties"]["answer"]["description"]
        assert beschreibung.startswith(erwartet), f"{sprache}: {beschreibung!r}"

# Demo Runbook
**Owner:** Obada Abdulhakim Kharaz — Project Manager & Demo Lead

This is the live presentation runbook for the ISU AI Chatbot final demo. Hard constraint: **total presentation time ≤ 10 minutes.**

---

## 1. Pre-Demo Checklist (do this before walking into the room)

- [ ] LM Studio is open, model loaded, local server started on port 1234 (or `OPENAI_API_KEY` set if using GPT-4o fallback)
- [ ] `chroma_db/` is populated — run `python rag/ingest_isu.py`, confirm `Done. Total chunks in database: 36`
- [ ] `streamlit run ui/app.py` launched, app reachable at `http://localhost:8501`
- [ ] Run one throwaway test question to warm up the model and confirm the pipeline works end-to-end
- [ ] Backup video (see `video_scripts.md`) is open in a second tab, paused at the start, ready to play if live demo fails
- [ ] Laptop is on charger / battery >50% — local LLM inference is power-hungry

---

## 2. Timing Breakdown (10 minutes total)

| Time | Segment | Speaker |
|---|---|---|
| 0:00–1:30 | Problem statement + why a chatbot (call-center load, response delay) | Obada |
| 1:30–3:00 | Architecture walkthrough (agents → RAG → safety → UI), using the diagram in the README | Zekeriya |
| 3:00–6:00 | **Live demo** — 3 sample questions asked in front of the room (see Section 3) | Obada |
| 6:00–7:30 | Safety monitoring explanation — show a flagged response and the risk score | Abdulaziz |
| 7:30–9:00 | Evaluation results — perplexity, task success rate, safety pass rate | Leen |
| 9:00–10:00 | Closing + questions | Full team |

If running over time, cut the third live-demo question first, then shorten the architecture walkthrough — never cut the live demo entirely; it's the strongest part of the pitch.

---

## 3. Live Demo Script — Questions to Ask (in order)

1. **"How do I register for courses on the OIS portal?"**
   Demonstrates grounded retrieval — answer should cite the OIS portal steps and the advisor requirement.
2. **"What is the shuttle schedule from Maslak?"**
   Demonstrates a narrow factual lookup with a short, clean answer.
3. **"What documents do I need for a residence permit?"**
   Demonstrates multi-fact retrieval (a list-style answer) and shows the safety badge passing on a longer response.

Each question should show: the question, the response, the agent label (`researcher`/`analyst`/`summarizer`), and the safety badge (✅ Passed).

---

## 4. Backup Plan — If the Live Demo Fails

Failure modes to expect: LM Studio not responding, slow inference (>60s) under room WiFi/laptop load, or a frozen Streamlit session.

**If the model is just slow:** narrate while waiting — explain the architecture diagram or safety pipeline out loud instead of standing in silence.

**If it actually breaks:** switch immediately to the backup video recording (see `video_scripts.md`) instead of debugging live. Do not attempt to fix code or restart services in front of the room — that costs more time than the 10-minute limit allows.

**Who controls the backup video:** Obada (Demo Lead) keeps it loaded and ready in a second browser tab for the entire presentation.

---

## 5. Post-Demo

- [ ] Confirm Q&A questions are logged for the report's "Discussion" section if any reveal a gap
- [ ] Note actual elapsed time against the 10-minute target for retrospective

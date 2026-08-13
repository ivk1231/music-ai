Music AI — Personal Apple Silicon Build

1. Copy "Music AI.app" into Applications.
2. Double-click it.
3. Choose an MP3/WAV and a result folder.
4. Leave the Small model selected for an M1 Air.
5. Leave instruments blank for automatic detection, or enter known groups such as:
   acoustic_piano,voice,drums,electric_bass,string_ensemble,flutes,violin
6. Choose the written meter if you know it. For 6/8, the app starts with an
   eighth-note pulse; change this only if the generated measure count is wrong.
7. Click Transcribe Music, then Open Score.

The app runs locally. It contains the personal Small transcription model and
beat model, so no GitHub or Hugging Face login is required.

If macOS blocks the app because it is a personal unsigned build:
Control-click Music AI.app, choose Open, then choose Open again.

Outputs:
- score.musicxml: editable notation for MuseScore or Sibelius
- arrangement.mid: multitrack performance data for a DAW or score app
- events.json: complete raw/derived machine data for later corrections

Known limitations:
- Instrument and meter detection are suggestions, not guarantees.
- Pickup/anacrusis export currently stops safely and asks for correction.
- Dense recordings can contain missing or extra notes.

// Imports
import { DataLoader } from "./data-loader.js";
import { SentenceGrouper, groupTokensIntoSentences } from "./sentence-grouper.js";
import { UIHandler } from "./ui-handler.js";
import { buildVocab } from './tokenizer.js';
import { trainLanguageModel } from './trainer.js';
import { loadTrainedModel } from './loadModel.js';
import { Predictor } from './predictor.js';

async function main() {
    // 💡 Zentrale Parameterdefinition
    const maxLen = 20;
    const embeddingDim = 64;
    const lstmUnits = 64;
    const epochs = 5;
    const batchSize = 16;

    // Initialisiere Klassen
    const loader = new DataLoader();
    const grouper = new SentenceGrouper();

    // Lade Daten
    const { trainData } = await loader.loadAll();

    // Gruppiere Tokens zu Sätzen
    const tokenGroups = groupTokensIntoSentences(trainData.map(d => d.token));
    console.log("Token-Gruppen (Sätze):", tokenGroups.slice(0, 3));

    // Vokabular aufbauen
    const vocab = buildVocab(tokenGroups);
    const unkCount = Object.entries(vocab).filter(([word, id]) => id === vocab["<UNK>"]).length;
    console.log("Vokabulargröße:", Object.keys(vocab).length);
    console.log("UNK-Zuweisungen im Vokabular:", unkCount);


    // ⏬ Versuche, ein bereits trainiertes Modell zu laden
    let model;
    try {
        model = await loadTrainedModel('trained-lm.json');
        console.log("✅ Vortrainiertes Modell geladen.");
    } catch (e) {
        console.warn("⚠️ Kein gespeichertes Modell gefunden. Du kannst eines trainieren.");
    }

    // Predictor instanziieren (ggf. mit Dummy-Modell – später ersetzt)
    const predictor = new Predictor(model, vocab, maxLen);

    // UI initialisieren
    const ui = new UIHandler(predictor);
    ui.init();

    // Training-Button
    document.getElementById('train-btn').addEventListener('click', async () => {
        const subset = tokenGroups.slice(0, 200); // Trainingsdaten

        // ⏬ Training starten mit konsistenten Parametern
        const trainedModel = await trainLanguageModel({
            tokenGroups: subset,
            vocab,
            maxLen,
            embeddingDim,
            lstmUnits,
            epochs,
            batchSize,
        });

        // Predictor aktualisieren
        predictor.setModel(trainedModel, vocab, maxLen);

        console.log("✅ Modelltraining abgeschlossen. Du kannst jetzt Texteingaben machen.");
    });
}

main();

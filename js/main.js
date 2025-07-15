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
    const lstmUnits = 128;
    const epochs = 20;
    const batchSize = 32;
    const sampleCount = 30000;

    // Initialisiere Klassen
    const loader = new DataLoader();

    // Lade Trainings- und Testdaten
    const { trainData } = await loader.loadAll();
    const { testData } = await loader.loadAll();

    // Gruppiere Tokens zu Sätzen (Train & Test)
    const fullTrainTokenGroups = groupTokensIntoSentences(trainData.map(d => d.token));

    analyzeSentenceLengths(fullTrainTokenGroups);
    console.log("Token-Gruppen (Train-Sätze):", fullTrainTokenGroups.slice(0, 3));

    // Vokabular aufbauen
    const trainTokenGroups = fullTrainTokenGroups.slice(0, sampleCount);
    const vocab = buildVocab(trainTokenGroups, 1, 10000); // Vokab auf Sample aufbauen
    console.log("Vokabulargröße:", Object.keys(vocab).length);
    const filteredTrainTokenGroups = filterToKnownTokens(trainTokenGroups, vocab);
    const testTokenGroups = groupTokensIntoSentences(testData.map(d => d.token));
    const filteredTestTokenGroups = filterToKnownTokens(testTokenGroups, vocab).slice(0, sampleCount);


    // ⏬ Vortrainiertes Modell laden (optional)
    let model;
    try {
        model = await loadTrainedModel();
        console.log("✅ Vortrainiertes Modell geladen.");
    } catch (e) {
        console.warn("⚠️ Kein gespeichertes Modell gefunden. Du kannst eines trainieren.");
    }

    // Predictor initialisieren
    const predictor = new Predictor(model, vocab, maxLen);

    // UI initialisieren
    const ui = new UIHandler(predictor);
    ui.init();

    // Training-Button
    document.getElementById('train-btn').addEventListener('click', async () => {

        // ⏬ Training starten (Trainer kümmert sich jetzt um das Preprocessing!)
        const trainedModel = await trainLanguageModel({
            tokenGroups: filteredTrainTokenGroups,
            valTokenGroups: filteredTestTokenGroups, // <== NEU: Raw Val-Gruppen übergeben
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

function analyzeSentenceLengths(tokenGroups) {
    const lengths = tokenGroups.map(s => s.length);
    const avg = (lengths.reduce((a, b) => a + b, 0) / lengths.length).toFixed(2);
    const median = lengths.sort((a, b) => a - b)[Math.floor(lengths.length / 2)];
    const shortCount = lengths.filter(l => l < 10).length;
    const longCount = lengths.filter(l => l > 50).length;
    const max = Math.max(...lengths);

    console.log("📏 Neue Satzlängen-Analyse:");
    console.log(`➡️ Durchschnitt: ${avg} Tokens`);
    console.log(`➡️ Median: ${median}`);
    console.log(`➡️ Maximum: ${max}`);
    console.log(`➡️ < 10 Tokens: ${shortCount} Sätze`);
    console.log(`➡️ > 50 Tokens: ${longCount} Sätze`);
}

main();

function filterToKnownTokens(tokenGroups, vocab) {
    return tokenGroups
        .map(tokens => tokens.filter(token => vocab.hasOwnProperty(token)))
        .filter(tokens => tokens.length > 1); // Nur sinnvolle Sequenzen behalten
}



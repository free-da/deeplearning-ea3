
// Imports
import { DataLoader } from "./data-loader.js";
import { SentenceGrouper, groupTokensIntoSentences } from "./sentence-grouper.js";
import { DummyPredictor } from "./predictor.js";
import { UIHandler } from "./ui-handler.js";
import { groupLabels, findBoundaries } from "./label-grouper.js";
import { labelToId, idToLabel } from './label-map.js';
import { buildVocab, tokensToSequences } from './tokenizer.js';
import { prepareLanguageModelData, padSequences } from './lm-preprocessing.js';



async function main() {
    // Initialisiere Klassen
    const loader = new DataLoader();
    const grouper = new SentenceGrouper();
    const predictor = new DummyPredictor();
    const ui = new UIHandler(predictor);

    // Lade Daten
    const { trainData, devData, testData } = await loader.loadAll();

    // Gruppiere Tokens zu Sätzen
    const tokenGroups = groupTokensIntoSentences(trainData.map(d => d.token));
    console.log("Token-Gruppen (Sätze):", tokenGroups.slice(0, 3));

    // Berechne Grenzen und gruppiere Labels entsprechend
    const labelBoundaries = findBoundaries(tokenGroups);
    const labels = trainData.map(d => d.label);
    const groupedLabels = groupLabels(labels, labelBoundaries);

    console.log("Label-to-ID:", labelToId);
    console.log("ID-to-Label:", idToLabel);
    console.log("Gruppierte Labels (Beispiel):", groupedLabels.slice(0, 3));

    // 👉 Vokabular aufbauen und Tokens in IDs umwandeln
    const vocab = buildVocab(tokenGroups);
    const tokenSequences = tokensToSequences(tokenGroups, vocab);
    // Padding auf eine einheitliche Länge, z. B. 50 Tokens
    const maxSeqLen = 50;
    const paddedTokenIdGroups = padSequences(tokenIdGroups, maxSeqLen);

    console.log("Vokabular-Größe:", Object.keys(vocab).length);
    console.log("Token-IDs (erste Sätze):", tokenSequences.slice(0, 2));
    console.log("Beispiel gepaddete Sequenz:", paddedTokenIdGroups[0]);

    // tokenIds: Array von Arrays mit Token-IDs pro Satz
    const { X, y } = prepareLanguageModelData(tokenIds);

    const maxLen = 20; // z. B. maximale Eingabelänge
    const X_padded = padSequences(X, maxLen);

    console.log("Beispiel Eingabe (padded):", X_padded.slice(0, 3));
    console.log("Zielwörter:", y.slice(0, 3));

    // Initialisiere Benutzeroberfläche
    ui.init();
}

main();

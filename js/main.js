// Imports
import { DataLoader } from "./data-loader.js";
import { SentenceGrouper, groupTokensIntoSentences } from "./sentence-grouper.js";
import { DummyPredictor } from "./predictor.js";
import { UIHandler } from "./ui-handler.js";
import { groupLabels, findBoundaries} from "./label-grouper.js";
import { labelToId, idToLabel } from './label-map.js';


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
    console.log("Token-Gruppen (Sätze):", tokenGroups.slice(0, 2));

    // Berechne Grenzen und gruppiere Labels entsprechend
    const labelBoundaries = findBoundaries(tokenGroups);
    const labels = trainData.map(d => d.label);
    const groupedLabels = groupLabels(labels, labelBoundaries);

    console.log(labelToId);
    console.log(idToLabel);

    console.log("Gruppierte Labels (Beispiel):", groupedLabels.slice(0, 3));

    // Initialisiere Benutzeroberfläche
    ui.init();
}

window.onload = main;

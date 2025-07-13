import {removeCitationsFromTokenGroups} from './lm-preprocessing.js';
import {SentenceGrouper} from "./sentence-grouper.js";
import {padSequences, tokensToSequences} from "./tokenizer.js";


export function preprocessTestData(testData, wordToId, maxLen) {
    const grouper = new SentenceGrouper();

    // 1. In Sätze gruppieren
    const tokenGroups = grouper.group(testData).map(g => g.map(x => x.token));

    // 2. Zitierungen entfernen
    const cleanedGroups = removeCitationsFromTokenGroups(tokenGroups);

    // 3. In ID-Sequenzen umwandeln
    const sequences = tokensToSequences(cleanedGroups, wordToId);

    // 4. Padding anwenden
    return padSequences(sequences, maxLen, wordToId['<PAD>']);

}

export function prepareLanguageModelData(sequences, vocabSize) {
    const X = [];
    const y = [];

    for (const seq of sequences) {
        for (let i = 1; i < seq.length; i++) {
            const inputSeq = seq.slice(0, i); // bis Token i (exkl.)
            const target = seq[i]; // vorherzusagendes Token

            X.push(inputSeq);
            y.push(target);
        }
    }

    return { X, y };
}

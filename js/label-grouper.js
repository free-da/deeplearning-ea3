// Gruppiert flache Labels entsprechend den gegebenen Token-Gruppen
export function groupLabels(flatLabels, boundaries) {
    const groupedLabels = [];

    for (const [start, end] of boundaries) {
        groupedLabels.push(flatLabels.slice(start, end));
    }

    return groupedLabels;
}

// Berechnet die Grenzen jeder Token-Gruppe (Start-/End-Index)
export function findBoundaries(tokenGroups) {
    const boundaries = [];
    let index = 0;

    for (const group of tokenGroups) {
        const length = group.length;
        boundaries.push([index, index + length]);
        index += length;
    }

    return boundaries;
}

// Erstellt ein Mapping von ID → Label
export function labelToIdDict(groupedLabels) {
    const labelSet = new Set();

    for (const group of groupedLabels) {
        for (const label of group) {
            labelSet.add(label);
        }
    }

    const sortedLabels = Array.from(labelSet).sort();
    const labelToId = {};

    sortedLabels.forEach((label, idx) => {
        labelToId[idx] = label;
    });

    return labelToId;
}

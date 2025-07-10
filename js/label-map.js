export const labelToId = {
    'O': 0,
    'B-DS': 1,
    'B-GP': 2,
    'B-OG': 3,
    'I-DS': 4,
    'I-GP': 5,
    'I-OG': 6,
};

export const idToLabel = Object.fromEntries(
    Object.entries(labelToId).map(([label, id]) => [id, label])
);

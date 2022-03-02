import {LocalStorage} from 'quasar'
export const BASE_URL = process.env.BASE_URL;
export const BASE_API_URL = `${BASE_URL}/rest-api`;

const BIDDERS_COLOR_MAPPING = {
  Alice: '#0000FF',
  Bob: '#FF0000',
  Charlie: '#abf561',
};

export default function () {
  return {
    config: {},
    experiments: [],
    runs: [],
    checkpoints: [],
    samples: [],
    games: [],
    responses: [],
    bidderColors: BIDDERS_COLOR_MAPPING,
    modelSelector: {},
    selector: {},
    // opening_explorer: {player: 0},
  };
}

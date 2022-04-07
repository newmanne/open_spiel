import {LocalStorage} from 'quasar'
export const BASE_URL = process.env.BASE_URL;
export const BASE_API_URL = `${BASE_URL}/rest-api`;

const BIDDERS_COLOR_MAPPING = {
  0: '#0000FF',
  1: '#FF0000',
  2: '#abf561',
  'Auctioneer': '#dfb100'
};

export default function () {
  return {
    config: {},
    experiments: [],
    samples: {},
    games: [],
    bidderColors: BIDDERS_COLOR_MAPPING,
  };
}

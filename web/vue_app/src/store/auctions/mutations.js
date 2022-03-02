import { range } from 'lodash';
import {LocalStorage} from 'quasar'
import Vue from 'vue'

export const API_FAILURE = (state, value) => {
    console.error(value)
  };

  export const SET_SAMPLES = (state, {data}) => {
    state.samples = data;
};

export const SET_GAMES = (state, {data}) => {
    state.games = data;
};

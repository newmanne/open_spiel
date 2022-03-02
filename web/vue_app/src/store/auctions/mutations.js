import { range } from 'lodash';
import {LocalStorage} from 'quasar'
import Vue from 'vue'

export const API_FAILURE = (state, value) => {
    console.error(value)
  };
  
export const SET_EXPERIMENTS = (state, {data, player}) => {
    state.modelSelector[player] = {
        experiments: data,
        runs: [],
        checkpoints: [],
        responses: [],
    };
};

export const SET_RUNS = (state, {data, player}) => {
    state.modelSelector[player].runs = data;
    state.modelSelector[player].checkpoints = [];
    state.modelSelector[player].responses = [];
};

export const SET_CHECKPOINTS = (state, {data, player}) => {
    state.modelSelector[player].checkpoints = data;
    state.modelSelector[player].responses = [];
};

export const SET_RESPONSES = (state, {data, player}) => {
    state.modelSelector[player].responses = data;
};
  
export const SET_SAMPLES = (state, {data}) => {
    state.samples = data;
};

export const SET_GAMES = (state, {data}) => {
    state.games = data;
};

export const SET_SELECTOR = (state, data) => {
    let {player, ...d} = data;
    state.selector[player] = d;
};

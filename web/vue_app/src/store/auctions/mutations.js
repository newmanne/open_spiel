import { range } from 'lodash';
import {LocalStorage} from 'quasar'
import Vue from 'vue'

export const API_FAILURE = (state, value) => {
    console.error(value)
  };

export const SET_SAMPLES = (state, {data}) => {
    state.samples = data;
};

export const ADD_SAMPLES_FROM_STATE = (state, {data}) => {
    state.samples_from_state.push(data.actions);
};

export const POP_SAMPLES_FROM_STATE = (state, {depth}) => {
    state.samples_from_state = state.samples_from_state.slice(0, depth);
}

export const SET_GAMES = (state, {data}) => {
    state.games = data;
};

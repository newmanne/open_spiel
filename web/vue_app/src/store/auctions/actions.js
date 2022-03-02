import { Notify } from 'quasar'
import { api } from 'boot/axios'

function _processError(commit, error, prefix = '') {
  switch (error.response.status) {
    case 404:
      Notify.notify({
        color: 'negative',
        message: prefix + error.response.data
      });
      break;
    case 401:
    //   commit('LOGOUT', true);
      // Vue.router.push({name: ''});
      break;
    case 500:
      commit('API_FAILURE', error);
    default:
      Notify({
        message: JSON.stringify(error),
        type: 'negative',
        closeBtn: true
      });
  }
  return false;
}

export const simpleGetter = (context, url, commit, params, commitParams) => {
    return api({
        method: 'GET', url: `${url}`, params: params,
      }).then(res => {
        if (commit != null) {
          context.commit(commit, {data: res.data, ...commitParams});
        }
        return res.data;
      }, error => {
        _processError(context.commit, error);
        return error;
      });
}

export const GET_EXPERIMENTS = (context) => {
    return simpleGetter(context, 'experiment/', 'SET_EXPERIMENTS');
};

export const GET_GAMES = (context) => {
  return simpleGetter(context, 'game/', 'SET_GAMES');
};

export const GET_RUNS = (context, {experimentPk}) => {
    return simpleGetter(context, `experiment/${experimentPk}/runs/`, 'SET_RUNS');
};
  
export const GET_CHECKPOINTS = (context, {runPk}) => {
  return simpleGetter(context, `run/${runPk}/checkpoints/`, 'SET_CHECKPOINTS');
};

export const GET_SAMPLES = (context, {gamePk, url_params}) => {
  return simpleGetter(context, `game/${gamePk}/samples/`, 'SET_SAMPLES', url_params)
};

export const GET_GAME_EXPERIMENTS = (context, {gamePk}) => {
  return simpleGetter(context, `game/${gamePk}/experiments`);
};

export const GET_GAME_RUNS = (context, {gamePk, experimentPk}) => {
  return simpleGetter(context, `game/${gamePk}/runs`, null, { experiment: experimentPk });
};

export const GET_RUN_CHECKPOINTS = (context, {runPk}) => {
  return simpleGetter(context, `run/${runPk}/checkpoints`);
};

export const GET_CHECKPOINT_RESPONSES = (context, {checkpointPk, player }) => {
  return simpleGetter(context, `checkpoint/${checkpointPk}/best_responses`, null, {player}, null);
};

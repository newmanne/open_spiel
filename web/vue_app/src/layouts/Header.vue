<template>
  <q-toolbar>
    <q-toolbar-title>
      Auction Strategy Explorer
      <small v-if="subtitle">
        {{subtitle}}
      </small>
    </q-toolbar-title>
    <q-tabs>
      <q-route-tab label="Opening Explorer" :to="{name: 'opening-explorer'}"/>
      <q-route-tab label="Strategy Explorer" :to="{name: 'strategy-explorer'}"/>
      <q-route-tab label="Cluster Explorer" :to="{name: 'cluster-explorer'}"/>
      <q-route-tab label="Run Trajectory" :to="{name: 'trajectory-plot'}"/>
      <q-route-tab label="Allocations" :to="{name: 'allocation-heatmap'}"/>
    </q-tabs>
  </q-toolbar>
</template>

<script>
  export default {
    name: "Header",
    props: ['subtitle'],
    methods: {
    }
  }
</script>

<style scoped>
</style>

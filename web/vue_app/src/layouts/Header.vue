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
      <!-- TODO: allocation viewer like 
      <q-route-tab label="Simulations" :to="{name: ''}"/>
      --> 
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

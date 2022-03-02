<template>
  <q-layout view="lHh Lpr lFf">
    <q-header elevated>
      <Header/>
    </q-header>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script>
  import { defineComponent, ref } from "vue";
  import Header from './Header'

  export default defineComponent({
    name: "MainLayout",

    components: {
      Header,
    },

    setup() {
    },
  });
</script>

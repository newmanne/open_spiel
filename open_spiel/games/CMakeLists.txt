set(GAME_SOURCES
  clock_auction.cc
  clock_auction.h
  backgammon.cc
  backgammon.h
  battleship.cc
  battleship.h
  battleship_types.h
  battleship_types.cc
  blackjack.cc
  blackjack.h
  blotto.cc
  blotto.h
  breakthrough.cc
  breakthrough.h
  bridge.cc
  bridge.h
  bridge/bridge_scoring.cc
  bridge/bridge_scoring.h
  bridge_uncontested_bidding.cc
  bridge_uncontested_bidding.h
  catch.cc
  catch.h
  chess.cc
  chess.h
  chess/chess_board.cc
  chess/chess_board.h
  chess/chess_common.cc
  chess/chess_common.h
  cliff_walking.cc
  cliff_walking.h
  clobber.cc
  clobber.h
  coin_game.cc
  coin_game.h
  connect_four.cc
  connect_four.h
  coop_box_pushing.cc
  coop_box_pushing.h
  coordinated_mp.cc
  coordinated_mp.h
  cursor_go.cc
  cursor_go.h
  dark_chess.cc
  dark_chess.h
  dark_hex.cc
  dark_hex.h
  deep_sea.cc
  deep_sea.h
  efg_game.cc
  efg_game.h
  efg_game_data.cc
  efg_game_data.h
  first_sealed_auction.cc
  first_sealed_auction.h
  gin_rummy.cc
  gin_rummy.h
  gin_rummy/gin_rummy_utils.cc
  gin_rummy/gin_rummy_utils.h
  go.cc
  go.h
  go/go_board.cc
  go/go_board.h
  goofspiel.cc
  goofspiel.h
  hallway.h
  hallway.cc
  havannah.cc
  havannah.h
  hearts.cc
  hearts.h
  hex.cc
  hex.h
  kriegspiel.cc
  kriegspiel.h
  kuhn_poker.cc
  kuhn_poker.h
  laser_tag.cc
  laser_tag.h
  leduc_poker.cc
  leduc_poker.h
  lewis_signaling.cc
  lewis_signaling.h
  liars_dice.cc
  liars_dice.h
  markov_soccer.cc
  markov_soccer.h
  matching_pennies_3p.cc
  matching_pennies_3p.h
  matrix_games.cc
  mfg/crowd_modelling.cc
  mfg/crowd_modelling.h
  mfg/crowd_modelling_2d.cc
  mfg/crowd_modelling_2d.h
  negotiation.cc
  negotiation.h
  nfg_game.cc
  nfg_game.h
  oh_hell.cc
  oh_hell.h
  oshi_zumo.cc
  oshi_zumo.h
  othello.cc
  othello.h
  oware.cc
  oware.h
  oware/oware_board.cc
  oware/oware_board.h
  pentago.cc
  pentago.h
  phantom_ttt.cc
  phantom_ttt.h
  pig.cc
  pig.h
  quoridor.cc
  quoridor.h
  rbc.cc
  rbc.h
  sheriff.cc
  sheriff.h
  skat.cc
  skat.h
  solitaire.cc
  solitaire.h
  stones_and_gems.cc
  stones_and_gems.h
  tarok.cc
  tarok.h
  tarok/cards.cc
  tarok/cards.h
  tarok/contracts.cc
  tarok/contracts.h
  tic_tac_toe.cc
  tic_tac_toe.h
  tiny_bridge.cc
  tiny_bridge.h
  tiny_hanabi.cc
  tiny_hanabi.h
  trade_comm.cc
  trade_comm.h
  y.cc
  y.h
)

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  set(GAME_SOURCES ${GAME_SOURCES} hanabi.cc hanabi.h)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  set(GAME_SOURCES ${GAME_SOURCES} universal_poker.cc universal_poker.h)
endif()

add_library (games OBJECT ${GAME_SOURCES})

target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (${OPEN_SPIEL_BUILD_WITH_HANABI})
  add_subdirectory(hanabi)
endif()
if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  add_subdirectory(universal_poker)
endif()
if (${OPEN_SPIEL_BUILD_WITH_GAMUT})
  add_subdirectory(gamut)
endif()


# Uncomment to build the Ludii demo
# add_subdirectory (ludii)

add_library(bridge_double_dummy_solver OBJECT
  bridge/double_dummy_solver/include/dll.h
  bridge/double_dummy_solver/include/portab.h
  bridge/double_dummy_solver/src/ABsearch.cpp
  bridge/double_dummy_solver/src/ABsearch.h
  bridge/double_dummy_solver/src/ABstats.cpp
  bridge/double_dummy_solver/src/ABstats.h
  bridge/double_dummy_solver/src/CalcTables.cpp
  bridge/double_dummy_solver/src/CalcTables.h
  bridge/double_dummy_solver/src/dds.cpp
  bridge/double_dummy_solver/src/dds.h
  bridge/double_dummy_solver/src/DealerPar.cpp
  bridge/double_dummy_solver/src/debug.h
  bridge/double_dummy_solver/src/dump.cpp
  bridge/double_dummy_solver/src/dump.h
  bridge/double_dummy_solver/src/File.cpp
  bridge/double_dummy_solver/src/File.h
  bridge/double_dummy_solver/src/Init.cpp
  bridge/double_dummy_solver/src/Init.h
  bridge/double_dummy_solver/src/LaterTricks.cpp
  bridge/double_dummy_solver/src/LaterTricks.h
  bridge/double_dummy_solver/src/Memory.cpp
  bridge/double_dummy_solver/src/Memory.h
  bridge/double_dummy_solver/src/Moves.cpp
  bridge/double_dummy_solver/src/Moves.h
  bridge/double_dummy_solver/src/Par.cpp
  bridge/double_dummy_solver/src/parallel.h
  bridge/double_dummy_solver/src/PBN.cpp
  bridge/double_dummy_solver/src/PBN.h
  bridge/double_dummy_solver/src/PlayAnalyser.cpp
  bridge/double_dummy_solver/src/PlayAnalyser.h
  bridge/double_dummy_solver/src/QuickTricks.cpp
  bridge/double_dummy_solver/src/QuickTricks.h
  bridge/double_dummy_solver/src/Scheduler.cpp
  bridge/double_dummy_solver/src/Scheduler.h
  bridge/double_dummy_solver/src/SolveBoard.cpp
  bridge/double_dummy_solver/src/SolveBoard.h
  bridge/double_dummy_solver/src/SolverIF.cpp
  bridge/double_dummy_solver/src/SolverIF.h
  bridge/double_dummy_solver/src/System.cpp
  bridge/double_dummy_solver/src/System.h
  bridge/double_dummy_solver/src/ThreadMgr.cpp
  bridge/double_dummy_solver/src/ThreadMgr.h
  bridge/double_dummy_solver/src/Timer.cpp
  bridge/double_dummy_solver/src/Timer.h
  bridge/double_dummy_solver/src/TimerGroup.cpp
  bridge/double_dummy_solver/src/TimerGroup.h
  bridge/double_dummy_solver/src/TimerList.cpp
  bridge/double_dummy_solver/src/TimerList.h
  bridge/double_dummy_solver/src/TimeStat.cpp
  bridge/double_dummy_solver/src/TimeStat.h
  bridge/double_dummy_solver/src/TimeStatList.cpp
  bridge/double_dummy_solver/src/TimeStatList.h
  bridge/double_dummy_solver/src/TransTable.h
  bridge/double_dummy_solver/src/TransTableL.cpp
  bridge/double_dummy_solver/src/TransTableL.h
  bridge/double_dummy_solver/src/TransTableS.cpp
  bridge/double_dummy_solver/src/TransTableS.h
)
target_include_directories (bridge_double_dummy_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bridge_double_dummy_solver PUBLIC DDS_NO_STATIC_INIT)

add_executable(clock_auction_test clock_auction_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(clock_auction_test clock_auction_test)

add_executable(crowd_modelling_test mfg/crowd_modelling_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(crowd_modelling_test crowd_modelling_test)

add_executable(crowd_modelling_2d_test mfg/crowd_modelling_2d_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(crowd_modelling_2d_test crowd_modelling_2d_test)

add_executable(rbc_test rbc_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(rbc_test rbc_test)


if (${OPEN_SPIEL_BUILD_WITH_ACPC})
  add_executable(universal_poker_test universal_poker_test.cc ${OPEN_SPIEL_OBJECTS}
          $<TARGET_OBJECTS:tests>
          $<TARGET_OBJECTS:algorithms>)
  add_test(universal_poker_test universal_poker_test
           --subgames_data_dir=${CMAKE_CURRENT_SOURCE_DIR}/universal_poker/endgames)
endif()

add_executable(y_test y_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(y_test y_test)


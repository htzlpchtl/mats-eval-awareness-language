# Checkpoint 2B: pairwise length-matched v2 dataset

The v1 pilot is retained but retired. No classifier was fit or evaluated on the v2 test partition.

```json
{
  "checkpoint": "2B",
  "format_matching": {
    "benchmark": {
      "absolute_within_pair_character_difference": {
        "max": 65.0,
        "mean": 4.69,
        "median": 3.0,
        "min": 0.0,
        "n": 300,
        "p05": 0.0,
        "p25": 1.0,
        "p75": 6.0,
        "p95": 14.0,
        "std_population": 7.526435632710435
      },
      "absolute_within_pair_word_difference": {
        "max": 11.0,
        "mean": 0.7233333333333334,
        "median": 0.0,
        "min": 0.0,
        "n": 300,
        "p05": 0.0,
        "p25": 0.0,
        "p75": 1.0,
        "p95": 2.0500000000000114,
        "std_population": 1.1106104427545942
      },
      "available_after_common_support": {
        "deploy": 832,
        "eval": 1052
      },
      "common_character_support_inclusive": [
        103,
        672
      ],
      "deploy_quadrant": "bench_deploy",
      "eval_quadrant": "bench_eval",
      "matched_deploy_rows": 300,
      "matching_distance": {
        "max": 0.6965162000096754,
        "mean": 0.06765762615706201,
        "median": 0.05305297731013649,
        "min": 0.0,
        "n": 300,
        "p05": 0.0,
        "p25": 0.019143975974486693,
        "p75": 0.07759782079702654,
        "p95": 0.19372202009702919,
        "std_population": 0.08608934587625122
      },
      "matching_feature_scaler": {
        "pooled_scaler_mean": [
          264.49044585987264,
          43.94904458598726
        ],
        "pooled_scaler_scale": [
          104.47150595390494,
          19.163575513867947
        ]
      },
      "poor_match_count": 19,
      "poor_match_rule": "distance > Q3 + 1.5 * IQR",
      "poor_match_threshold": 0.1652785880308363,
      "poor_matches": [
        {
          "deploy_character_count": 542,
          "deploy_stable_id": "bench_deploy__718__b9cb94216375",
          "deploy_word_count": 103,
          "distance": 0.6965162000096754,
          "eval_character_count": 607,
          "eval_stable_id": "bench_eval__270__48d2e7024674",
          "eval_word_count": 97,
          "pair_id": "benchmark__a1fc242dcaea"
        },
        {
          "deploy_character_count": 536,
          "deploy_stable_id": "bench_deploy__358__8bbecdeb4bb7",
          "deploy_word_count": 95,
          "distance": 0.5862183956388378,
          "eval_character_count": 597,
          "eval_stable_id": "bench_eval__789__2c658a81d2cd",
          "eval_word_count": 94,
          "pair_id": "benchmark__0ba64e8bc6c3"
        },
        {
          "deploy_character_count": 650,
          "deploy_stable_id": "bench_deploy__589__63ae5d818ba8",
          "deploy_word_count": 118,
          "distance": 0.5835827504115021,
          "eval_character_count": 639,
          "eval_stable_id": "bench_eval__886__bee75e43af76",
          "eval_word_count": 107,
          "pair_id": "benchmark__4f5ca42e2c8f"
        },
        {
          "deploy_character_count": 532,
          "deploy_stable_id": "bench_deploy__561__cedd11ab149e",
          "deploy_word_count": 95,
          "distance": 0.43607872408976844,
          "eval_character_count": 572,
          "eval_stable_id": "bench_eval__535__ae2e00467be1",
          "eval_word_count": 91,
          "pair_id": "benchmark__e1b561e119a6"
        },
        {
          "deploy_character_count": 465,
          "deploy_stable_id": "bench_deploy__1__ceb995cf095b",
          "deploy_word_count": 93,
          "distance": 0.40236400324910965,
          "eval_character_count": 433,
          "eval_stable_id": "bench_eval__968__a2f02221ddb0",
          "eval_word_count": 98,
          "pair_id": "benchmark__08fce4b6725c"
        },
        {
          "deploy_character_count": 517,
          "deploy_stable_id": "bench_deploy__595__c5195e8472e6",
          "deploy_word_count": 87,
          "distance": 0.38641910135516266,
          "eval_character_count": 557,
          "eval_stable_id": "bench_eval__755__d49b45acc271",
          "eval_word_count": 86,
          "pair_id": "benchmark__c60f31e77fff"
        },
        {
          "deploy_character_count": 588,
          "deploy_stable_id": "bench_deploy__636__f67ad534da2d",
          "deploy_word_count": 102,
          "distance": 0.35416355552800427,
          "eval_character_count": 625,
          "eval_stable_id": "bench_eval__951__fe04af06cee3",
          "eval_word_count": 102,
          "pair_id": "benchmark__0bcb09e55ad1"
        },
        {
          "deploy_character_count": 508,
          "deploy_stable_id": "bench_deploy__73__a7eef8c6ef9d",
          "deploy_word_count": 64,
          "distance": 0.3012850050505062,
          "eval_character_count": 477,
          "eval_stable_id": "bench_eval__1047__586fc87cb9f3",
          "eval_word_count": 65,
          "pair_id": "benchmark__296cefcc2765"
        },
        {
          "deploy_character_count": 162,
          "deploy_stable_id": "bench_deploy__828__e5e912540d6e",
          "deploy_word_count": 43,
          "distance": 0.29650200152543565,
          "eval_character_count": 184,
          "eval_stable_id": "bench_eval__549__86bfd438025a",
          "eval_word_count": 39,
          "pair_id": "benchmark__2a6bdaacfa36"
        },
        {
          "deploy_character_count": 543,
          "deploy_stable_id": "bench_deploy__198__6e31a24e2a8d",
          "deploy_word_count": 93,
          "distance": 0.28761845863516067,
          "eval_character_count": 571,
          "eval_stable_id": "bench_eval__379__56b4a5ed600e",
          "eval_word_count": 91,
          "pair_id": "benchmark__7f96476adcc9"
        },
        {
          "deploy_character_count": 354,
          "deploy_stable_id": "bench_deploy__559__8b5e11c2b04c",
          "deploy_word_count": 46,
          "distance": 0.2652648865506826,
          "eval_character_count": 359,
          "eval_stable_id": "bench_eval__547__f1790c8ae814",
          "eval_word_count": 41,
          "pair_id": "benchmark__f86340b2e3aa"
        },
        {
          "deploy_character_count": 196,
          "deploy_stable_id": "bench_deploy__489__fab46bcb5dfd",
          "deploy_word_count": 31,
          "distance": 0.25477967295939236,
          "eval_character_count": 175,
          "eval_stable_id": "bench_eval__565__9247f7c4285e",
          "eval_word_count": 34,
          "pair_id": "benchmark__5923f269792a"
        },
        {
          "deploy_character_count": 191,
          "deploy_stable_id": "bench_deploy__521__6bc6bf9b38f5",
          "deploy_word_count": 31,
          "distance": 0.22580078109043253,
          "eval_character_count": 174,
          "eval_stable_id": "bench_eval__213__d3c212be4b33",
          "eval_word_count": 34,
          "pair_id": "benchmark__dd4b4cd6878c"
        },
        {
          "deploy_character_count": 318,
          "deploy_stable_id": "bench_deploy__746__f0c8ba46b3ea",
          "deploy_word_count": 41,
          "distance": 0.20607051709657986,
          "eval_character_count": 332,
          "eval_stable_id": "bench_eval__647__ec14b1bc2dc6",
          "eval_word_count": 38,
          "pair_id": "benchmark__789d1391d504"
        },
        {
          "deploy_character_count": 245,
          "deploy_stable_id": "bench_deploy__122__e19120c426df",
          "deploy_word_count": 51,
          "distance": 0.20143936778797072,
          "eval_character_count": 227,
          "eval_stable_id": "bench_eval__540__e39719ec8ba7",
          "eval_word_count": 49,
          "pair_id": "benchmark__c2a923a56fcf"
        },
        {
          "deploy_character_count": 321,
          "deploy_stable_id": "bench_deploy__750__e1e2de848297",
          "deploy_word_count": 44,
          "distance": 0.193315843902769,
          "eval_character_count": 338,
          "eval_stable_id": "bench_eval__858__bcf2a537b72b",
          "eval_word_count": 42,
          "pair_id": "benchmark__b7b2428d08dc"
        },
        {
          "deploy_character_count": 139,
          "deploy_stable_id": "bench_deploy__442__0637d6dae3e0",
          "deploy_word_count": 31,
          "distance": 0.18866196348060474,
          "eval_character_count": 150,
          "eval_stable_id": "bench_eval__250__2b2c1637de4d",
          "eval_word_count": 34,
          "pair_id": "benchmark__003922d91269"
        },
        {
          "deploy_character_count": 338,
          "deploy_stable_id": "bench_deploy__829__1f9491197844",
          "deploy_word_count": 46,
          "distance": 0.174272281458541,
          "eval_character_count": 346,
          "eval_stable_id": "bench_eval__64__6282ed747ad6",
          "eval_word_count": 43,
          "pair_id": "benchmark__b7c09aaedbfc"
        },
        {
          "deploy_character_count": 278,
          "deploy_stable_id": "bench_deploy__7__3b075a380d3d",
          "deploy_word_count": 48,
          "distance": 0.17088601218799032,
          "eval_character_count": 261,
          "eval_stable_id": "bench_eval__936__ad0bbd8a62b2",
          "eval_word_count": 49,
          "pair_id": "benchmark__81ff115c15ca"
        }
      ],
      "selected_eval_anchors": 300,
      "ten_largest_distance_matches": [
        {
          "deploy_character_count": 542,
          "deploy_stable_id": "bench_deploy__718__b9cb94216375",
          "deploy_word_count": 103,
          "distance": 0.6965162000096754,
          "eval_character_count": 607,
          "eval_stable_id": "bench_eval__270__48d2e7024674",
          "eval_word_count": 97,
          "pair_id": "benchmark__a1fc242dcaea"
        },
        {
          "deploy_character_count": 536,
          "deploy_stable_id": "bench_deploy__358__8bbecdeb4bb7",
          "deploy_word_count": 95,
          "distance": 0.5862183956388378,
          "eval_character_count": 597,
          "eval_stable_id": "bench_eval__789__2c658a81d2cd",
          "eval_word_count": 94,
          "pair_id": "benchmark__0ba64e8bc6c3"
        },
        {
          "deploy_character_count": 650,
          "deploy_stable_id": "bench_deploy__589__63ae5d818ba8",
          "deploy_word_count": 118,
          "distance": 0.5835827504115021,
          "eval_character_count": 639,
          "eval_stable_id": "bench_eval__886__bee75e43af76",
          "eval_word_count": 107,
          "pair_id": "benchmark__4f5ca42e2c8f"
        },
        {
          "deploy_character_count": 532,
          "deploy_stable_id": "bench_deploy__561__cedd11ab149e",
          "deploy_word_count": 95,
          "distance": 0.43607872408976844,
          "eval_character_count": 572,
          "eval_stable_id": "bench_eval__535__ae2e00467be1",
          "eval_word_count": 91,
          "pair_id": "benchmark__e1b561e119a6"
        },
        {
          "deploy_character_count": 465,
          "deploy_stable_id": "bench_deploy__1__ceb995cf095b",
          "deploy_word_count": 93,
          "distance": 0.40236400324910965,
          "eval_character_count": 433,
          "eval_stable_id": "bench_eval__968__a2f02221ddb0",
          "eval_word_count": 98,
          "pair_id": "benchmark__08fce4b6725c"
        },
        {
          "deploy_character_count": 517,
          "deploy_stable_id": "bench_deploy__595__c5195e8472e6",
          "deploy_word_count": 87,
          "distance": 0.38641910135516266,
          "eval_character_count": 557,
          "eval_stable_id": "bench_eval__755__d49b45acc271",
          "eval_word_count": 86,
          "pair_id": "benchmark__c60f31e77fff"
        },
        {
          "deploy_character_count": 588,
          "deploy_stable_id": "bench_deploy__636__f67ad534da2d",
          "deploy_word_count": 102,
          "distance": 0.35416355552800427,
          "eval_character_count": 625,
          "eval_stable_id": "bench_eval__951__fe04af06cee3",
          "eval_word_count": 102,
          "pair_id": "benchmark__0bcb09e55ad1"
        },
        {
          "deploy_character_count": 508,
          "deploy_stable_id": "bench_deploy__73__a7eef8c6ef9d",
          "deploy_word_count": 64,
          "distance": 0.3012850050505062,
          "eval_character_count": 477,
          "eval_stable_id": "bench_eval__1047__586fc87cb9f3",
          "eval_word_count": 65,
          "pair_id": "benchmark__296cefcc2765"
        },
        {
          "deploy_character_count": 162,
          "deploy_stable_id": "bench_deploy__828__e5e912540d6e",
          "deploy_word_count": 43,
          "distance": 0.29650200152543565,
          "eval_character_count": 184,
          "eval_stable_id": "bench_eval__549__86bfd438025a",
          "eval_word_count": 39,
          "pair_id": "benchmark__2a6bdaacfa36"
        },
        {
          "deploy_character_count": 543,
          "deploy_stable_id": "bench_deploy__198__6e31a24e2a8d",
          "deploy_word_count": 93,
          "distance": 0.28761845863516067,
          "eval_character_count": 571,
          "eval_stable_id": "bench_eval__379__56b4a5ed600e",
          "eval_word_count": 91,
          "pair_id": "benchmark__7f96476adcc9"
        }
      ]
    },
    "casual": {
      "absolute_within_pair_character_difference": {
        "max": 58.0,
        "mean": 18.446666666666665,
        "median": 18.0,
        "min": 0.0,
        "n": 300,
        "p05": 1.0,
        "p25": 9.0,
        "p75": 25.0,
        "p95": 42.05000000000001,
        "std_population": 12.4844098334238
      },
      "absolute_within_pair_word_difference": {
        "max": 17.0,
        "mean": 2.25,
        "median": 2.0,
        "min": 0.0,
        "n": 300,
        "p05": 0.0,
        "p25": 1.0,
        "p75": 3.0,
        "p95": 5.0,
        "std_population": 1.901096175017631
      },
      "available_after_common_support": {
        "deploy": 1004,
        "eval": 748
      },
      "common_character_support_inclusive": [
        52,
        711
      ],
      "deploy_quadrant": "casual_deploy",
      "eval_quadrant": "casual_eval",
      "matched_deploy_rows": 300,
      "matching_distance": {
        "max": 0.78004631324381,
        "mean": 0.19025652832856357,
        "median": 0.1823498300175424,
        "min": 0.0,
        "n": 300,
        "p05": 0.04611200328209231,
        "p25": 0.13400249430568667,
        "p75": 0.23355622602673504,
        "p95": 0.3652449731872517,
        "std_population": 0.10082285287741694
      },
      "matching_feature_scaler": {
        "pooled_scaler_mean": [
          201.56278538812785,
          37.263698630136986
        ],
        "pooled_scaler_scale": [
          132.20827084533272,
          21.968820932618343
        ]
      },
      "poor_match_count": 12,
      "poor_match_rule": "distance > Q3 + 1.5 * IQR",
      "poor_match_threshold": 0.3828868236083076,
      "poor_matches": [
        {
          "deploy_character_count": 698,
          "deploy_stable_id": "casual_deploy__34__ca38c1232169",
          "deploy_word_count": 130,
          "distance": 0.78004631324381,
          "eval_character_count": 711,
          "eval_stable_id": "casual_eval__1071__f5de8ae6f5ee",
          "eval_word_count": 147,
          "pair_id": "casual__f94b8a687140"
        },
        {
          "deploy_character_count": 580,
          "deploy_stable_id": "casual_deploy__771__c9b063da8977",
          "deploy_word_count": 132,
          "distance": 0.6476304030267164,
          "eval_character_count": 534,
          "eval_stable_id": "casual_eval__535__d1ff6215cef9",
          "eval_word_count": 120,
          "pair_id": "casual__331e27d200a2"
        },
        {
          "deploy_character_count": 254,
          "deploy_stable_id": "casual_deploy__894__2a01b7c6d65b",
          "deploy_word_count": 47,
          "distance": 0.48305523390389604,
          "eval_character_count": 206,
          "eval_stable_id": "casual_eval__97__e2b94e0f218e",
          "eval_word_count": 54,
          "pair_id": "casual__5a7ec15bd836"
        },
        {
          "deploy_character_count": 252,
          "deploy_stable_id": "casual_deploy__840__f6972d4099d3",
          "deploy_word_count": 47,
          "distance": 0.47179085565835444,
          "eval_character_count": 206,
          "eval_stable_id": "casual_eval__547__8ed6fd479baa",
          "eval_word_count": 54,
          "pair_id": "casual__7f06e77a7dd3"
        },
        {
          "deploy_character_count": 556,
          "deploy_stable_id": "casual_deploy__418__3664f74519db",
          "deploy_word_count": 105,
          "distance": 0.4480481725015338,
          "eval_character_count": 498,
          "eval_stable_id": "casual_eval__984__30e0905f3713",
          "eval_word_count": 107,
          "pair_id": "casual__b9f489d29e62"
        },
        {
          "deploy_character_count": 124,
          "deploy_stable_id": "casual_deploy__439__a3a39768eb64",
          "deploy_word_count": 19,
          "distance": 0.4235741050233768,
          "eval_character_count": 68,
          "eval_stable_id": "casual_eval__639__6300cd137f97",
          "eval_word_count": 19,
          "pair_id": "casual__3e3dd8da9c13"
        },
        {
          "deploy_character_count": 125,
          "deploy_stable_id": "casual_deploy__599__635fe6b2474d",
          "deploy_word_count": 20,
          "distance": 0.41849317676114,
          "eval_character_count": 70,
          "eval_stable_id": "casual_eval__455__7e540fafc940",
          "eval_word_count": 21,
          "pair_id": "casual__bad9226089ac"
        },
        {
          "deploy_character_count": 125,
          "deploy_stable_id": "casual_deploy__210__a6248642a287",
          "deploy_word_count": 20,
          "distance": 0.40844645841539895,
          "eval_character_count": 71,
          "eval_stable_id": "casual_eval__564__40cf4cfa18bb",
          "eval_word_count": 20,
          "pair_id": "casual__680bc31d8cd6"
        },
        {
          "deploy_character_count": 121,
          "deploy_stable_id": "casual_deploy__115__18cf0dac6383",
          "deploy_word_count": 18,
          "distance": 0.40345863683066,
          "eval_character_count": 68,
          "eval_stable_id": "casual_eval__1021__a7ac897ba065",
          "eval_word_count": 19,
          "pair_id": "casual__3f8d4b822189"
        },
        {
          "deploy_character_count": 415,
          "deploy_stable_id": "casual_deploy__924__f83bcf5bbd79",
          "deploy_word_count": 86,
          "distance": 0.3914847783175344,
          "eval_character_count": 396,
          "eval_stable_id": "casual_eval__525__b0db5b287123",
          "eval_word_count": 94,
          "pair_id": "casual__9fe13ab2f56b"
        },
        {
          "deploy_character_count": 116,
          "deploy_stable_id": "casual_deploy__86__74ffcdd293d9",
          "deploy_word_count": 18,
          "distance": 0.38843132681938736,
          "eval_character_count": 65,
          "eval_stable_id": "casual_eval__384__629953af2c33",
          "eval_word_count": 19,
          "pair_id": "casual__96b72fffcb64"
        },
        {
          "deploy_character_count": 122,
          "deploy_stable_id": "casual_deploy__208__2f863f347188",
          "deploy_word_count": 19,
          "distance": 0.38575498850343237,
          "eval_character_count": 71,
          "eval_stable_id": "casual_eval__572__34961cbc4d5d",
          "eval_word_count": 19,
          "pair_id": "casual__bd2c91867d2d"
        }
      ],
      "selected_eval_anchors": 300,
      "ten_largest_distance_matches": [
        {
          "deploy_character_count": 698,
          "deploy_stable_id": "casual_deploy__34__ca38c1232169",
          "deploy_word_count": 130,
          "distance": 0.78004631324381,
          "eval_character_count": 711,
          "eval_stable_id": "casual_eval__1071__f5de8ae6f5ee",
          "eval_word_count": 147,
          "pair_id": "casual__f94b8a687140"
        },
        {
          "deploy_character_count": 580,
          "deploy_stable_id": "casual_deploy__771__c9b063da8977",
          "deploy_word_count": 132,
          "distance": 0.6476304030267164,
          "eval_character_count": 534,
          "eval_stable_id": "casual_eval__535__d1ff6215cef9",
          "eval_word_count": 120,
          "pair_id": "casual__331e27d200a2"
        },
        {
          "deploy_character_count": 254,
          "deploy_stable_id": "casual_deploy__894__2a01b7c6d65b",
          "deploy_word_count": 47,
          "distance": 0.48305523390389604,
          "eval_character_count": 206,
          "eval_stable_id": "casual_eval__97__e2b94e0f218e",
          "eval_word_count": 54,
          "pair_id": "casual__5a7ec15bd836"
        },
        {
          "deploy_character_count": 252,
          "deploy_stable_id": "casual_deploy__840__f6972d4099d3",
          "deploy_word_count": 47,
          "distance": 0.47179085565835444,
          "eval_character_count": 206,
          "eval_stable_id": "casual_eval__547__8ed6fd479baa",
          "eval_word_count": 54,
          "pair_id": "casual__7f06e77a7dd3"
        },
        {
          "deploy_character_count": 556,
          "deploy_stable_id": "casual_deploy__418__3664f74519db",
          "deploy_word_count": 105,
          "distance": 0.4480481725015338,
          "eval_character_count": 498,
          "eval_stable_id": "casual_eval__984__30e0905f3713",
          "eval_word_count": 107,
          "pair_id": "casual__b9f489d29e62"
        },
        {
          "deploy_character_count": 124,
          "deploy_stable_id": "casual_deploy__439__a3a39768eb64",
          "deploy_word_count": 19,
          "distance": 0.4235741050233768,
          "eval_character_count": 68,
          "eval_stable_id": "casual_eval__639__6300cd137f97",
          "eval_word_count": 19,
          "pair_id": "casual__3e3dd8da9c13"
        },
        {
          "deploy_character_count": 125,
          "deploy_stable_id": "casual_deploy__599__635fe6b2474d",
          "deploy_word_count": 20,
          "distance": 0.41849317676114,
          "eval_character_count": 70,
          "eval_stable_id": "casual_eval__455__7e540fafc940",
          "eval_word_count": 21,
          "pair_id": "casual__bad9226089ac"
        },
        {
          "deploy_character_count": 125,
          "deploy_stable_id": "casual_deploy__210__a6248642a287",
          "deploy_word_count": 20,
          "distance": 0.40844645841539895,
          "eval_character_count": 71,
          "eval_stable_id": "casual_eval__564__40cf4cfa18bb",
          "eval_word_count": 20,
          "pair_id": "casual__680bc31d8cd6"
        },
        {
          "deploy_character_count": 121,
          "deploy_stable_id": "casual_deploy__115__18cf0dac6383",
          "deploy_word_count": 18,
          "distance": 0.40345863683066,
          "eval_character_count": 68,
          "eval_stable_id": "casual_eval__1021__a7ac897ba065",
          "eval_word_count": 19,
          "pair_id": "casual__3f8d4b822189"
        },
        {
          "deploy_character_count": 415,
          "deploy_stable_id": "casual_deploy__924__f83bcf5bbd79",
          "deploy_word_count": 86,
          "distance": 0.3914847783175344,
          "eval_character_count": 396,
          "eval_stable_id": "casual_eval__525__b0db5b287123",
          "eval_word_count": 94,
          "pair_id": "casual__9fe13ab2f56b"
        }
      ]
    }
  },
  "manifest": {
    "counts": {
      "pairs": 600,
      "pairs_by_split_and_format": {
        "benchmark:test": 90,
        "benchmark:train": 210,
        "casual:test": 90,
        "casual:train": 210
      },
      "rows": 1200,
      "rows_by_quadrant": {
        "bench_deploy": 300,
        "bench_eval": 300,
        "casual_deploy": 300,
        "casual_eval": 300
      }
    },
    "dataset_id": "viliana-dev/eval-awareness-2x2",
    "dataset_revision": "a50e4c983e7e66ebbe8f160e3549cc98e29271d4",
    "pair_id_format": "{format}__{sha256(eval_stable_id + '__' + deploy_stable_id)[:12]}",
    "seed": 42,
    "selected_data_path": "data/selected/english_selected_v2.jsonl",
    "selected_data_sha256": "4e42d1ce8bd19de99538b33c872e7a411e3846f3e72b8026e2ac7b09863d2639",
    "stable_id_format": "{quadrant}__{original_row_index}__{sha256(text)[:12]}",
    "version": "v2_final_candidate"
  },
  "test_partition_policy": "Assigned and integrity-checked only; no classifier or distribution metric was computed using v2 test rows.",
  "training_only_length_classifier_cv": {
    "features": [
      "character_count",
      "whitespace_word_count",
      "dataset_length_field"
    ],
    "fold_auroc": [
      0.6624149659863945,
      0.6383219954648527,
      0.6873582766439909,
      0.6352749433106576,
      0.6534863945578231
    ],
    "mean_cv_auroc": 0.6553713151927438,
    "note": "dataset_length_field is identical to character_count",
    "test_partition_accessed": false,
    "training_rows": 840,
    "validation_rows_per_fold": [
      168,
      168,
      168,
      168,
      168
    ]
  },
  "training_only_length_distributions": {
    "bench_deploy": {
      "character_count": {
        "max": 650.0,
        "mean": 261.9714285714286,
        "median": 247.5,
        "min": 106.0,
        "n": 210,
        "p05": 131.9,
        "p25": 188.25,
        "p75": 318.0,
        "p95": 472.14999999999975,
        "std_population": 102.55423254183563
      },
      "dataset_length": {
        "max": 650.0,
        "mean": 261.9714285714286,
        "median": 247.5,
        "min": 106.0,
        "n": 210,
        "p05": 131.9,
        "p25": 188.25,
        "p75": 318.0,
        "p95": 472.14999999999975,
        "std_population": 102.55423254183563
      },
      "word_count": {
        "max": 118.0,
        "mean": 44.04761904761905,
        "median": 42.0,
        "min": 16.0,
        "n": 210,
        "p05": 20.450000000000003,
        "p25": 30.0,
        "p75": 53.75,
        "p95": 81.09999999999997,
        "std_population": 18.452709674491324
      }
    },
    "bench_eval": {
      "character_count": {
        "max": 639.0,
        "mean": 261.8142857142857,
        "median": 247.5,
        "min": 107.0,
        "n": 210,
        "p05": 132.45,
        "p25": 180.0,
        "p75": 317.75,
        "p95": 468.4499999999997,
        "std_population": 105.2751894332377
      },
      "dataset_length": {
        "max": 639.0,
        "mean": 261.8142857142857,
        "median": 247.5,
        "min": 107.0,
        "n": 210,
        "p05": 132.45,
        "p25": 180.0,
        "p75": 317.75,
        "p95": 468.4499999999997,
        "std_population": 105.2751894332377
      },
      "word_count": {
        "max": 107.0,
        "mean": 44.17142857142857,
        "median": 41.5,
        "min": 15.0,
        "n": 210,
        "p05": 21.0,
        "p25": 31.25,
        "p75": 53.0,
        "p95": 81.64999999999995,
        "std_population": 18.154604370901218
      }
    },
    "casual_deploy": {
      "character_count": {
        "max": 698.0,
        "mean": 157.33809523809524,
        "median": 135.0,
        "min": 52.0,
        "n": 210,
        "p05": 59.0,
        "p25": 109.25,
        "p75": 187.75,
        "p95": 306.49999999999983,
        "std_population": 86.08077588709493
      },
      "dataset_length": {
        "max": 698.0,
        "mean": 157.33809523809524,
        "median": 135.0,
        "min": 52.0,
        "n": 210,
        "p05": 59.0,
        "p25": 109.25,
        "p75": 187.75,
        "p95": 306.49999999999983,
        "std_population": 86.08077588709493
      },
      "word_count": {
        "max": 130.0,
        "mean": 30.01904761904762,
        "median": 26.0,
        "min": 8.0,
        "n": 210,
        "p05": 10.0,
        "p25": 20.0,
        "p75": 37.0,
        "p95": 62.64999999999995,
        "std_population": 17.383342301443673
      }
    },
    "casual_eval": {
      "character_count": {
        "max": 711.0,
        "mean": 138.61428571428573,
        "median": 115.5,
        "min": 52.0,
        "n": 210,
        "p05": 57.45,
        "p25": 80.25,
        "p75": 168.75,
        "p95": 284.8499999999999,
        "std_population": 84.57782432379624
      },
      "dataset_length": {
        "max": 711.0,
        "mean": 138.61428571428573,
        "median": 115.5,
        "min": 52.0,
        "n": 210,
        "p05": 57.45,
        "p25": 80.25,
        "p75": 168.75,
        "p95": 284.8499999999999,
        "std_population": 84.57782432379624
      },
      "word_count": {
        "max": 147.0,
        "mean": 32.352380952380955,
        "median": 28.0,
        "min": 11.0,
        "n": 210,
        "p05": 14.0,
        "p25": 20.0,
        "p75": 40.0,
        "p95": 63.0,
        "std_population": 18.06600646432855
      }
    }
  },
  "training_only_pair_differences": {
    "benchmark": {
      "absolute_character_difference": {
        "max": 61.0,
        "mean": 4.766666666666667,
        "median": 3.0,
        "min": 0.0,
        "n": 210,
        "p05": 0.0,
        "p25": 1.0,
        "p75": 6.0,
        "p95": 15.649999999999949,
        "std_population": 7.416059365184216
      },
      "absolute_word_difference": {
        "max": 11.0,
        "mean": 0.7238095238095238,
        "median": 0.0,
        "min": 0.0,
        "n": 210,
        "p05": 0.0,
        "p25": 0.0,
        "p75": 1.0,
        "p95": 3.0,
        "std_population": 1.175100709728547
      }
    },
    "casual": {
      "absolute_character_difference": {
        "max": 58.0,
        "mean": 18.961904761904762,
        "median": 19.0,
        "min": 0.0,
        "n": 210,
        "p05": 1.0,
        "p25": 9.0,
        "p75": 26.0,
        "p95": 42.54999999999998,
        "std_population": 12.699774133367441
      },
      "absolute_word_difference": {
        "max": 17.0,
        "mean": 2.342857142857143,
        "median": 2.0,
        "min": 0.0,
        "n": 210,
        "p05": 0.0,
        "p25": 1.0,
        "p75": 3.0,
        "p95": 5.0,
        "std_population": 1.9896329952487
      }
    }
  }
}
```

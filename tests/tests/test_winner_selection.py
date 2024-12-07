from collections import defaultdict
from operator import itemgetter
from unittest import TestCase

from base.checkpoint import Uid, Key
from weight_setting.winner_selection import get_contestant_ranks, calculate_rank_weights


class WinnerSelectionTest(TestCase):
    def test_winner_selection(self):
        winners: dict[Key, list[Uid]] = defaultdict(list)

        for uid, data in SAMPLE_CONTEST_RESULTS.items():
            scores = {key: info[0] for key, info in data.items()}
            submitted_blocks = {key: info[1] for key, info in data.items()}

            ranks = get_contestant_ranks(scores)
            weights = calculate_rank_weights(submitted_blocks, ranks)

            winner = max(
                weights.items(),
                key=itemgetter(1),
                default=None
            )

            if not winner:
                continue

            winners[winner[0]].append(uid)

        msg = "Multiple winners found:\n" + "\n".join(
            f"{winner}: {', '.join(map(str, validator_uids))}"
            for winner, validator_uids in winners.items()
        )

        self.assertEqual(len(winners), 1, msg=msg)


SAMPLE_CONTEST_RESULTS = {
    32: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5G8kx5jy4pWCFHs2SykNkWLKgfdL6ishH4MaWCR1XgWzffVW": (0.0047350296800949794, 4349867),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.03750760643421349, 4318177),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.03843751265234748, 4306607),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.038466702547769786, 4391977),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.0388877829812723, 4346536),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.04128225244370703, 4346263),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.04130371824113756, 4346301),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.041822735400599255, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.048702321208805695, 4346572),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.08523826212616632, 4333813),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.08607464340963819, 4341940),

        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.12901275625547776, 4344899),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.1302997308442186, 4325834),
        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.13038968834587714, 4391951),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.13096671366557996, 4366450),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.1317781783405036, 4344903),
        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.13210417523885512, 4344888),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.13234495140442123, 4337132),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.13239411278488092, 4344893),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.1331300964430438, 4338175),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.13333990870363904, 4395412),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.13344030423383316, 4395355),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.13357812487549273, 4388244),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.133979967546254, 4372844),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.13411732813466898, 4372816),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.13486338689252023, 4395410),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.1352037469249406, 4386752),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.1357543991230612, 4372868),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.1367277712699381, 4395416),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.1368528832580907, 4380869),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.1369128971453332, 4395353),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.13703739856604374, 4380871),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.13740284977146117, 4382392),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.13764635221853358, 4380867),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.13788964875115012, 4382388),
    },
    39: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.04435761097342165, 4318177),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.04555665030381639, 4306607),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.04694017180880022, 4346301),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.0469944652091518, 4346536),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.047167598894511886, 4346263),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.04732959122561632, 4391977),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.04739131820795417, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.05421845868145197, 4346572),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.0844636994673934, 4333813),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.08477220593500888, 4341940),

        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.1491838084896248, 4391951),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.1492964187254551, 4372844),
        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.14994699695588906, 4344888),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.15153486960513127, 4344903),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.1516340495313512, 4338175),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.15166038147386546, 4388244),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.15180483651592963, 4344893),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.15217363441928752, 4337132),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.15275758891638433, 4366450),
        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.15382495095673104, 4344899),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.15417833298483602, 4395410),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.15441447264771832, 4325834),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.15450568005169693, 4395355),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.15559867867426688, 4395412),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.15622103984049504, 4372868),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.156560199629361, 4386752),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.1570233427370304, 4372816),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.15786815477658464, 4380869),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.15899502650089328, 4382388),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.16054258442085895, 4395416),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.16077389312146323, 4395353),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.16100518453681115, 4380871),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.16135470668513976, 4380867),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.1615705823532001, 4382392),
    },
    96: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.044223579546008106, 4318177),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.04505110715041962, 4346536),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.04513620514496535, 4391977),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.04520716062013907, 4346263),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.04521907869207647, 4346301),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.04537230756205225, 4306607),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.04591360460066118, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.052569116502875816, 4346572),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.08572450048819245, 4341940),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.08575811109364528, 4333813),

        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.15529239058807365, 4391951),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.1562509481466541, 4338175),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.15658708337984123, 4344893),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.15679319314343473, 4337132),
        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.15705641408640836, 4344888),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.1571840921693975, 4344903),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.15749861178397043, 4366450),
        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.158519204340923, 4344899),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.1586975624917107, 4395412),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.15899496579534336, 4325834),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.15918900709285255, 4395410),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.1591893347784979, 4395355),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.15986905494537457, 4372868),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.159886345088734, 4388244),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.16001797441025373, 4372816),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.16012395440396712, 4372844),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.16077151246618854, 4386752),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.16521143306207503, 4380867),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.1661624830994251, 4380869),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.1662188962181843, 4395416),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.1664902137221375, 4382388),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.16653723406772805, 4395353),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.16659025746490955, 4382392),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.1667230393617234, 4380871),
    },
    121: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5G8kx5jy4pWCFHs2SykNkWLKgfdL6ishH4MaWCR1XgWzffVW": (0.01254088770851263, 4349867),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.04241387855774764, 4318177),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.043746581425091115, 4391977),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.04374962988944463, 4306607),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.04393193051886336, 4346536),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.04566212902787164, 4346263),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.04594338763977211, 4346301),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.04636463127101, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.053016137446198146, 4346572),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.08770633448228753, 4333813),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.0882223155240078, 4341940),

        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.15236631621294047, 4344888),
        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.15305119626176422, 4391951),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.15357785178664818, 4344903),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.15403955270431074, 4344893),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.1563782477934783, 4338175),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.1564700963416261, 4337132),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.15648988483201515, 4372844),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.15717508228962498, 4395410),
        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.15719920836525508, 4344899),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.15725746475780705, 4388244),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.1574479353690953, 4366450),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.15757708373477858, 4325834),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.15782234180529908, 4395412),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.15847364722680762, 4372868),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.15864558277911336, 4395355),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.15998785115593073, 4386752),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.16046119796359443, 4372816),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.16359290829798037, 4382388),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.16374269812597486, 4395353),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.1643033391473441, 4380871),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.1650214933484178, 4380869),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.1657146592364935, 4395416),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.16592750836069062, 4380867),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.16650367692093965, 4382392),
    },
    144: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5G8kx5jy4pWCFHs2SykNkWLKgfdL6ishH4MaWCR1XgWzffVW": (0.004727407281399506, 4349867),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.04230336821546489, 4318177),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.0436331779589973, 4391977),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.04367372479088351, 4346536),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.04376467863477784, 4306607),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.04557739478401628, 4346263),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.04592640390111796, 4346301),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.04643442116195825, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.053099062913233575, 4346572),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.08168411036533038, 4341940),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.08312816104635508, 4333813),

        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.13732835026742207, 4391951),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.14112584752186288, 4366450),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.14239991268099308, 4337132),
        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.14313284740645824, 4344888),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.14484275592389448, 4372868),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.145000292762477, 4338175),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.14503333603998936, 4386752),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.1457091871327679, 4344903),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.1462505539828472, 4388244),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.1465202871096825, 4372816),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.1475525985717073, 4372844),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.14897518269202342, 4395355),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.15051652829511772, 4395412),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.1506977306432623, 4380869),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.15359808389078752, 4380871),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.15402456426667013, 4382388),
        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.1562290913603015, 4344899),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.156497582487481, 4344893),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.1596157737865311, 4325834),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.16030434996807147, 4395410),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.16034602497942457, 4380867),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.16314602372700712, 4395416),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.16371213493648823, 4382392),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.16741466759701942, 4395353),
    },
    251: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5G8kx5jy4pWCFHs2SykNkWLKgfdL6ishH4MaWCR1XgWzffVW": (0.006220607498596889, 4349867),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.037944029449805065, 4318177),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.03906486819092797, 4391977),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.03909525985761915, 4306607),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.039338926941620035, 4346536),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.04166641216689661, 4346301),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.04191032286995429, 4346263),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.042207807313304195, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.0490862326143518, 4346572),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.08739224629470223, 4341940),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.08763651057836275, 4333813),

        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.13050730237517952, 4391951),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.13172709884124592, 4366450),
        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.13210687329406462, 4344899),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.13234625282438853, 4325834),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.13317049938759568, 4338175),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.13330849539974698, 4344903),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.13334546943306844, 4337132),
        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.13338069013657042, 4344888),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.13352536688664543, 4344893),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.1346933525158672, 4395355),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.1347386960222487, 4395412),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.13564601993575717, 4386752),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.13567425294830568, 4372816),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.13568047965859314, 4395410),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.13572869519824493, 4372844),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.13575491329047065, 4372868),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.13597216869920634, 4388244),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.13857444592391285, 4380871),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.1386065192180041, 4382388),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.13872452946407124, 4382392),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.1387326414265691, 4395416),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.13879920152081832, 4380869),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.13880015533424744, 4380867),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.13885832258836545, 4395353),
    },
    254: {
        "5C5S27czwoKAeTUaVyBEdRvgWkFrqMmgbPK8UBx6hZyUNTeh": (0.0, 4352777),
        "5Cfwy342AJRhii1BbJfGarRkjDpBFaCDjbpYLoqJbs7G8CKf": (0.0, 4344880),
        "5Gj6drDauh4kMq1PSVRXFV3rQ1Sdyj4oeSfzQ6uc3Zb1aarA": (0.0, 4318183),
        "5G8kx5jy4pWCFHs2SykNkWLKgfdL6ishH4MaWCR1XgWzffVW": (0.005836890633434564, 4349867),
        "5D9vocHwLH5p5R1QNtqverMR1MLWHT1p7e6818rfBdJTyvQ5": (0.03827949133671798, 4318177),
        "5FgZ45kjsK9NpQ2kj5FzQpLLJASnLt3y5Q4xASB584r8fqYN": (0.03925965739170691, 4306607),
        "5FeiMAAGw5exSCYyUupNsyNPoqqPwpMUJ2KTmDBc9N8LMHdn": (0.03928339151968509, 4391977),
        "5FFPF5Ys5LSrMgDy4cuxRbXHHZJTeHdLGHgNbzJSjYpTRuzK": (0.039725460470916296, 4346536),
        "5GhGoCnWpShtPDfZFqKVDsWEfV6mfFbJ5f2CQSXjAfyNUWZx": (0.042007816300506574, 4346301),
        "5HKpi9BTM6HaZE1v6pnbRKFuEVZryu3HQYFPrUKYV2WUF7SC": (0.04225400307314187, 4346263),
        "5FQpDGb7SEXpQrBPQ3pydFybbf1taHG6GneasGPBS2iiPpwJ": (0.04254518935349413, 4346462),
        "5FH3MuamXgMW6hNG4F7WkvFfT634CTWw7vAVgDsVHtViGpbj": (0.049422286789766526, 4346572),
        "5E2Mg7CfNxYeSXEkNeKoGJovU5ZuZRbqQ9dXu4t5QBh4KrG6": (0.08530837814274014, 4333813),
        "5EA7ANRy37ptapTzR2Ysi6832HdgniLvmNBB4kT7ZLdkyiCT": (0.08629943600238267, 4341940),

        "5Hda5ab3WKCDpCTtFkTNxtFvSWwrWuZaWvha4PdMWqGgf8Qt": (0.13472270441250597, 4391951),
        "5GdziaHAvFEfByztoLWXcT2RsNd27Y5MihDQtmGGRnYbLzES": (0.13495637618182638, 4366450),
        "5GeXrw6zgjmQNjPb2fxFDdPoRWw6QJuxHdmDECmEk4ckNJCo": (0.13515966844641394, 4344899),
        "5DSVkr3yJbKsLpxYrd3yEwWqeKHFTDkT9aNAJ3BGVweZzVNz": (0.13528824059417535, 4344903),
        "5GYvFFLrdZ3qripesRu8QcFQCZ5f1PEJXUTqDCw23wADrGXf": (0.13554812185678458, 4325834),
        "5Do3NMXaPAs34UURXKfKGP9CFcD78ZtuHFVTbkvDqnnuUcM3": (0.13607711487855526, 4337132),
        "5CDPaSSyVhZFcxTrgeF7YyTRoSYmsVLZPk6CRrZPCFskkKFH": (0.1363275480363926, 4338175),
        "5CChRf9PEwhbtEVgr3d48TrK42TEDS21AVNBUs5CtBDiMXci": (0.13652778223687603, 4344893),
        "5F6zvqCNXtbHuqo2arEpHXWY6sPYafZtXtnocXzdTDWU69je": (0.13657230834019415, 4344888),
        "5HKJyXpfBLVfmYCd1FurPhNx1ySbRDo8ZLkNvx88CX2HKquN": (0.13770681035488672, 4395412),
        "5FbqAsuoCtiihyDGEGDiZ28jftUcQ6QPC2SLYH18xY5SECwE": (0.13798292101337503, 4395355),
        "5G9jnB1xz6UdYWBDtsi6rqAXFfrujyHsY7ftjRYnAxvPFdHX": (0.13813924959807092, 4372844),
        "5FsjFsz5M4nzMWYY64aQm6QtEpHxGfQfzrAanoPRCX2dvyAS": (0.1386309749144908, 4388244),
        "5ERbEqu1GfXXNuQaLCSTz5C2gTPUgkjUwYzvHzsKBDLnfxqa": (0.13873104235812525, 4395410),
        "5FEZHdPQshPC9jiCztP9T68va3c2oRgJnsjSXxWseZRtDBn3": (0.1396363870591968, 4386752),
        "5FvSjcxo8ouVrHsiPyYS4qbswtNAcocRGdEKyVGw1qe3J6KT": (0.13964664639825622, 4372816),
        "5CiM9BdoX4m1w8N35B46Lni6odNxfSQJgxVkSVHuK57o56qE": (0.1397067078511609, 4372868),
        "5CcvDYEPV2ofeMFYXV5qGoyzpcRsYmRyVAAdxYwH6eH7AdZE": (0.141726268993051, 4395416),
        "5CdLcRFL5CRJjMPoAC65CyX2kk4MNkqZLZ1g5b1Cn7vpL3o3": (0.14181206178065356, 4380869),
        "5FLRV3hYybfKQAuUW4woC7XJVHd16zJDgFTWn53drVeoLJcH": (0.1419026581282685, 4382392),
        "5GRFwhrRPDz3ksFkq2mbtr4pkbfNQofN3K7sHUTiGq6vjYNw": (0.1420794841797523, 4380867),
        "5CZxfaKehsyD1PiPUqc285KihVw3eD8WG9o5Gnb6bbCfcq11": (0.14213391908848336, 4382388),
        "5CFKDdvrs4Up8oAbgmQjRMhMGcg4Atn4YfPGav7CYBRV4Sp4": (0.1421340712226904, 4380871),
        "5D7JqYHHcyCJCMVrhKfbarYEq2pLVG1U7mpbiQDQK5YQ6GRm": (0.14213888286563478, 4395353),
    },
}
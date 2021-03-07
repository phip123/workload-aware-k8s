import random

from srds import ParameterizedDistribution as PDist

# python-pi was made with req=10000, except xeongpu - this one was req=2000, due to the inexplicable slow response
execution_time_distributions = {
    ('rpi3', 'faas-workloads/python-pi'): (1.909987211227417, 2.203219413757324, PDist.lognorm(
        ((1.5561351906586467,), 1.9092154433285176, 0.03832589013392289))),
    ('rpi4', 'faas-workloads/python-pi'): (1.6961045265197754, 1.7755048274993896, PDist.lognorm(
        ((2.0378973843923234,), 1.6960468751471511, 0.017276257000211737))),
    ('xeongpu', 'faas-workloads/python-pi'): (2.9558749198913574, 3.0182230472564697, PDist.lognorm(
        ((0.33924794244178996,), 2.9421626090307855, 0.03231526504686262))),
    ('xeoncpu', 'faas-workloads/python-pi'): (2.9558749198913574, 3.0182230472564697, PDist.lognorm(
        ((0.33924794244178996,), 2.9421626090307855, 0.03231526504686262))),
    ('nx', 'faas-workloads/python-pi'): (1.280782699584961, 1.3314659595489502, PDist.lognorm(
        ((0.3204243885534094,), 1.2658584067446368, 0.029294830565182426))),
    ('tx2', 'faas-workloads/python-pi'): (0.9472050666809082, 0.973700761795044, PDist.lognorm(
        ((0.34123939468038056,), 0.9407173896320605, 0.015806120791955418))),
    ('nano', 'faas-workloads/python-pi'): (1.300102949142456, 1.330549716949463, PDist.lognorm(
        ((0.7189304098250027,), 1.2990304403954829, 0.007775231273223189))),
    ('rockpi', 'faas-workloads/python-pi'): (0.9023468494415283, 0.96213698387146, PDist.lognorm(
        ((1.0176060175917105,), 0.9019280787708568, 0.008452707474514693))),
    ('coral', 'faas-workloads/python-pi'): (1.6157212257385254, 1.6432366371154785, PDist.lognorm(
        ((2.0685603472046896,), 1.6156998209501152, 0.0066422665082161245))),
    ('nuc', 'faas-workloads/python-pi'): (0.4291870594024658, 0.44258952140808105, PDist.lognorm(
        ((0.7668200520309387,), 0.42879374181896945, 0.002728658224322277))),
    ('xeongpu', 'faas-workloads/fio'): (1.1168184280395508, 1.1966462135314941, PDist.lognorm(
        ((0.547478970870047,), 1.1083822085143797, 0.030433853662662387))),
    ('xeoncpu', 'faas-workloads/fio'): (1.1168184280395508, 1.1966462135314941, PDist.lognorm(
        ((0.547478970870047,), 1.1083822085143797, 0.030433853662662387))),
    ('tx2', 'faas-workloads/fio'): (4.199761152267456, 4.508903503417969,
                                    PDist.lognorm(((1.3756415929126355,), 4.198414125199262, 0.05290299893253018))),
    ('nano', 'faas-workloads/fio'): (17.634405612945557, 21.379503965377808,
                                     PDist.lognorm(((0.018126581624838864,), -20.429942918329502, 40.06238600923392))),
    ('nx', 'faas-workloads/fio'): (13.390366792678833, 14.499114036560059,
                                   PDist.lognorm(((1.2593738429645396,), 13.379751217809101, 0.21565860828690103))),
    ('rpi4', 'faas-workloads/fio'): (23.43714690208435, 33.209991216659546,
                                     PDist.lognorm(((0.009384653682239431,), -156.36010377296788, 184.15597782535895))),
    ('nuc', 'faas-workloads/fio'): (1.0893218517303467, 1.3713059425354004,
                                    PDist.lognorm(((1.1299582738854954,), 1.088922789515207, 0.014824511843598303))),
    ('rockpi', 'faas-workloads/fio'): (21.46393871307373, 23.428807735443115,
                                       PDist.lognorm(((0.6243415046474259,), 21.333338896210595, 0.544673148530165))),
    ('rpi3', 'faas-workloads/fio'): (23.947848558425903, 41.15871047973633,
                                     PDist.lognorm(((1.4725990652984735,), 23.901942788072812, 2.061301749245649))),
    ('tx2', 'faas-workloads/resnet-inference-cpu'): (0.6948370933532715, 1.1444413661956787, PDist.lognorm(
        ((0.8501001928589622,), 0.6917307756410416, 0.03637932277634568))),
    ('nano', 'faas-workloads/resnet-inference-cpu'): (0.8841347694396973, 1.3210082054138184, PDist.lognorm(
        ((0.6418412926282235,), 0.8774528367440888, 0.040203191672859434))),
    ('nx', 'faas-workloads/resnet-inference-cpu'): (0.4536397457122803, 0.5972743034362793, PDist.lognorm(
        ((0.18877396743289954,), 0.36851360580284376, 0.13846196638192304))),
    ('rpi3', 'faas-workloads/resnet-inference-cpu'): (
        4.425604581832886, 5.861693859100342,
        PDist.lognorm(((0.480548812316585,), 4.109022481450841, 0.8837760799244128))),
    ('rpi4', 'faas-workloads/resnet-inference-cpu'): (2.772036075592041, 3.4497311115264893, PDist.lognorm(
        ((0.34259117242902337,), 2.6647105511948013, 0.22908468382108765))),
    ('rockpi', 'faas-workloads/resnet-inference-cpu'): (1.1331660747528076, 1.9165077209472656, PDist.lognorm(
        ((0.1701387081476361,), 0.6767214411001377, 0.6915067219689892))),
    ('coral', 'faas-workloads/resnet-inference-cpu'): (1.4113075733184814, 3.642624855041504, PDist.lognorm(
        ((0.5091677176621551,), 1.1979548997737532, 0.8673575947926709))),
    ('xeongpu', 'faas-workloads/resnet-inference-cpu'): (0.1462233066558838, 0.3092348575592041, PDist.lognorm(
        ((1.250516246833203,), 0.14569116166838314, 0.015314005461484962))),
    ('xeoncpu', 'faas-workloads/resnet-inference-cpu'): (0.1462233066558838, 0.3092348575592041, PDist.lognorm(
        ((1.250516246833203,), 0.14569116166838314, 0.015314005461484962))),
    ('nuc', 'faas-workloads/resnet-inference-cpu'): (0.1407608985900879, 0.30925583839416504, PDist.lognorm(
        ((0.5323003236246481,), 0.1365508992652769, 0.017830791693920288))),
    ('tx2', 'faas-workloads/resnet-inference-gpu'): (0.37734198570251465, 0.8128652572631836, PDist.lognorm(
        ((0.6929061829413992,), 0.37615968175516823, 0.012486450479634974))),
    ('nano', 'faas-workloads/resnet-inference-gpu'): (0.6658952236175537, 1.843451738357544, PDist.lognorm(
        ((0.9361873291014332,), 0.6630484909941858, 0.04025457729806213))),
    ('nx', 'faas-workloads/resnet-inference-gpu'): (0.36667823791503906, 0.7085371017456055, PDist.lognorm(
        ((0.6224342740828939,), 0.36467625323618025, 0.017596239652913598))),
    ('xeongpu', 'faas-workloads/resnet-inference-gpu'): (0.11229228973388672, 0.3232553005218506, PDist.lognorm(
        ((0.8643766901903747,), 0.11095425513074908, 0.011781111260569395))),
    ('tx2', 'faas-workloads/speech-inference-gpu'): (3.1765449047088623, 3.380650758743286, PDist.lognorm(
        ((0.0025414734025261208,), -9.223028569212083, 12.529918252908715))),
    ('nano', 'faas-workloads/speech-inference-gpu'): (4.365040063858032, 4.694480657577515, PDist.lognorm(
        ((0.013090577085142809,), -1.5754467886453463, 6.119082935261158))),
    ('nx', 'faas-workloads/speech-inference-gpu'): (1.5870838165283203, 1.835263967514038, PDist.lognorm(
        ((0.3409027343356294,), 1.5597613802886927, 0.08327712822735661))),
    ('xeongpu', 'faas-workloads/speech-inference-gpu'): (0.71323561668396, 0.7919847965240479, PDist.lognorm(
        ((0.008946086677742587,), -1.307494985069547, 2.055274866615258))),
    ('tx2', 'faas-workloads/speech-inference-tflite'): (3.3613061904907227, 3.548738956451416, PDist.lognorm(
        ((0.14837049015268372,), 3.1825127376318507, 0.2542879539682151))),
    ('nano', 'faas-workloads/speech-inference-tflite'): (3.811147451400757, 4.026976585388184, PDist.lognorm(
        ((0.338573110489491,), 3.756233885601591, 0.12222610657663688))),
    ('nx', 'faas-workloads/speech-inference-tflite'): (2.511276960372925, 2.8111839294433594, PDist.lognorm(
        ((0.002128915799735443,), -20.11768232046066, 22.801283873362763))),
    ('rpi3', 'faas-workloads/speech-inference-tflite'): (15.982834100723267, 17.307713270187378, PDist.lognorm(
        ((0.2990468329948087,), 15.530564025376192, 0.9842449631740879))),
    ('rpi4', 'faas-workloads/speech-inference-tflite'): (6.625596523284912, 6.853733539581299, PDist.lognorm(
        ((0.013426059217138844,), 3.1072194637679287, 3.6418182858862655))),
    ('rockpi', 'faas-workloads/speech-inference-tflite'): (3.65761399269104, 3.972121000289917, PDist.lognorm(
        ((0.0038685069906899405,), -10.314587245344875, 14.13031293229767))),
    ('xeongpu', 'faas-workloads/speech-inference-tflite'): (1.0549635887145996, 1.1173889636993408, PDist.lognorm(
        ((0.3862192323604978,), 1.040493169321096, 0.03485261244686297))),
    ('xeoncpu', 'faas-workloads/speech-inference-tflite'): (1.0549635887145996, 1.1173889636993408, PDist.lognorm(
        ((0.3862192323604978,), 1.040493169321096, 0.03485261244686297))),
    ('coral', 'faas-workloads/speech-inference-tflite'): (
        7.140416860580444, 7.32071328163147,
        PDist.lognorm(((0.309383084086699,), 7.075666750430653, 0.13197037140529588))),
    ('nuc', 'faas-workloads/speech-inference-tflite'): (1.0520174503326416, 1.086385726928711, PDist.lognorm(
        ((0.7164044762741202,), 1.0506221929357897, 0.009455495987769815))),
    ('tx2', 'faas-workloads/mobilenet-inference-tflite'): (0.3218262195587158, 0.3404886722564697, PDist.lognorm(
        ((2.533344881031916,), 0.32182288558187044, 0.0042115822256853265))),
    ('nano', 'faas-workloads/mobilenet-inference-tflite'): (0.4417688846588135, 0.45539116859436035, PDist.lognorm(
        ((2.23565414673075,), 0.44176341795738083, 0.002910299253376422))),
    ('nx', 'faas-workloads/mobilenet-inference-tflite'): (0.3183910846710205, 0.35461950302124023, PDist.lognorm(
        ((0.28469241141582113,), 0.3091602236680885, 0.022375250553106865))),
    ('rpi3', 'faas-workloads/mobilenet-inference-tflite'): (
        1.96744704246521, 2.3119328022003174,
        PDist.lognorm(((0.402693282261331,), 1.9085815209567794, 0.151517311826892))),
    ('rpi4', 'faas-workloads/mobilenet-inference-tflite'): (1.2563717365264893, 1.3392689228057861, PDist.lognorm(
        ((0.492750480136671,), 1.2433487914150247, 0.03528135900856))),
    ('rockpi', 'faas-workloads/mobilenet-inference-tflite'): (0.37113523483276367, 0.6942205429077148, PDist.lognorm(
        ((0.15694893846245875,), 0.09924997518597733, 0.411072865085738))),
    ('coral', 'faas-workloads/mobilenet-inference-tflite'): (0.6539947986602783, 0.7023055553436279, PDist.lognorm(
        ((0.48206879938366154,), 0.6520314724778564, 0.011274621134062308))),
    ('xeongpu', 'faas-workloads/mobilenet-inference-tflite'): (0.26646852493286133, 0.3171541690826416, PDist.lognorm(
        ((0.5365853735187458,), 0.2629530210923786, 0.017526290870830248))),
    ('xeoncpu', 'faas-workloads/mobilenet-inference-tflite'): (0.26646852493286133, 0.3171541690826416, PDist.lognorm(
        ((0.5365853735187458,), 0.2629530210923786, 0.017526290870830248))),
    ('nuc', 'faas-workloads/mobilenet-inference-tflite'): (0.2647244930267334, 0.308530330657959, PDist.lognorm(
        ((0.213495159031093,), 0.24391558677812142, 0.03560607539324892))),
    ('coral', 'faas-workloads/mobilenet-inference-tpu'): (0.531743049621582, 0.5755233764648438, PDist.lognorm(
        ((0.23466525119207204,), 0.5059029828658818, 0.0405198874588389))),
    ('tx2', 'faas-workloads/resnet-preprocessing'): (6.223581314086914, 6.590674638748169, PDist.lognorm(
        ((0.25278510391411824,), 6.076221026439702, 0.30137273301561684))),
    ('nano', 'faas-workloads/resnet-preprocessing'): (7.8491973876953125, 9.16739535331726, PDist.lognorm(
        ((0.749277313721395,), 7.838747661086966, 0.07855304435000107))),
    ('nx', 'faas-workloads/resnet-preprocessing'): (5.809688329696655, 6.8749473094940186, PDist.lognorm(
        ((0.2463529274947488,), 5.5980717469353, 0.46803545566088733))),
    ('rpi3', 'faas-workloads/resnet-preprocessing'): (
        29.41605019569397, 37.47779178619385,
        PDist.lognorm(((0.5964107022199113,), 29.14633433496855, 1.104606368132186))),
    ('rpi4', 'faas-workloads/resnet-preprocessing'): (18.986950397491455, 20.01262331008911, PDist.lognorm(
        ((0.0070302901079828135,), -17.918047627124665, 37.41970089494406))),
    ('rockpi', 'faas-workloads/resnet-preprocessing'): (7.352227449417114, 8.141456127166748, PDist.lognorm(
        ((0.1897235590296447,), 6.987563394961493, 0.6473314423433307))),
    ('coral', 'faas-workloads/resnet-preprocessing'): (10.412590026855469, 10.664041996002197, PDist.lognorm(
        ((0.14905146429856836,), 10.169515023380942, 0.34020180035649983))),
    ('nuc', 'faas-workloads/resnet-preprocessing'): (2.467944383621216, 2.5904436111450195, PDist.lognorm(
        ((0.21058554606390106,), 2.4047184516326245, 0.11860319137639974))),
    ('xeongpu', 'faas-workloads/resnet-preprocessing'): (2.5816872119903564, 2.73801851272583, PDist.lognorm(
        ((0.0046462923279184595,), -3.733773151864244, 6.398168798011675))),
    ('xeoncpu', 'faas-workloads/resnet-preprocessing'): (2.5816872119903564, 2.73801851272583, PDist.lognorm(
        ((0.0046462923279184595,), -3.733773151864244, 6.398168798011675))),
    ('xeongpu', 'faas-workloads/resnet-training-gpu'): (
        31.53477692604065, 33.14484930038452,
        PDist.lognorm(((0.438998700689963,), 31.14152001046308, 0.8976046157724586))),
    ('nx', 'faas-workloads/resnet-training-gpu'): (139.35805344581604, 144.51621437072754, PDist.lognorm(
        ((0.005832169839304721,), -51.585196234614244, 193.5819207725017))),
    ('tx2', 'faas-workloads/resnet-training-gpu'): (225.4308431148529, 234.05239510536194, PDist.lognorm(
        ((0.6361116541813518,), 224.70438940974208, 2.810213847353603))),
    ('nano', 'faas-workloads/resnet-training-gpu'): (
        475.3121247291565, 3758.72234249115,
        PDist.lognorm(((1.3586403533926732,), 468.0979297934913, 139.09890480835355))),
    ('nuc', 'faas-workloads/resnet-training-cpu'): (196.37506127357483, 202.84338760375977,
                                                    PDist.lognorm(
                                                        ((0.4738306908521043,), 196.1424565418967, 1.153017771389426))),
    ('nx', 'faas-workloads/tf-gpu'): (1.158355951309204, 1.208902359008789,
                                      PDist.lognorm(((0.6352649591792237,), 1.1545879504235148, 0.015347490963527024))),
    ('xeongpu', 'faas-workloads/tf-gpu'): (0.36005401611328125, 0.37538766860961914, PDist.lognorm(
        ((2.5307323293089565,), 0.3600503383478125, 0.004224701754546815))),
    ('tx2', 'faas-workloads/tf-gpu'): (1.8746130466461182, 2.1380443572998047,
                                       PDist.lognorm(((0.6207149378318482,), 1.873085227543282, 0.011399394443365792))),
    ('nano', 'faas-workloads/tf-gpu'): (0.5251214504241943, 1.096886157989502, PDist.lognorm(
        ((1.4344624279117903,), 0.5247122204199652, 0.03296898373982664))),
}


def get_resnet_fet(device_type, mode):
    interval = None
    if device_type == 'cloudvm':
        if mode == 'gpu':
            interval = (140, 200)
        if mode == 'cpu':
            interval = (260, 300)
    if device_type == 'coral':
        if mode == 'cpu':
            interval = (1000, 1060)
        if mode == 'tpu':
            interval = (60, 70)
    if device_type == 'nuc':
        interval = (500, 550)
    if device_type == 'rockpi':
        interval = (700, 1000)
    if device_type == 'rpi3':
        interval = (1400, 1600)
    if device_type == 'rpi4':
        interval = (1000, 1200)
    if device_type == 'server':
        if mode == 'gpu':
            interval = (100, 150)
        if mode == 'cpu':
            interval = (220, 260)
    if device_type == 'tx2':
        if mode == 'gpu':
            interval = (120, 160)
        if mode == 'cpu':
            interval = (420, 660)
    try:
        return random.randint(*interval) / 1000
    except TypeError:
        return None


#         print(device_type, mode)

min_max_execution_times = {'faas-workloads/tf-gpu': (0.36531192779541016, 1.8881448245048522),
                           'faas-workloads/speech-inference-gpu': (0.747898964881897, 4.544314980506897),
                           'faas-workloads/resnet-training-gpu': (32.12711091739376, 847.171815255109),
                           'faas-workloads/speech-inference-tflite': (1.0626225304603576, 16.559413411617278),
                           'faas-workloads/resnet-preprocessing': (2.5259750509262084, 30.478582429331404),
                           'faas-workloads/mobilenet-inference-tflite': (0.2803403902053833, 2.0728497886657715),
                           'faas-workloads/python-pi': (0.24590346574783326, 71.5935873889923),
                           'faas-workloads/fio': (1.1227234315872192, 28.66214724727299),
                           'faas-workloads/resnet-inference-cpu': (0.15748303413391113, 5.095058534145355),
                           'faas-workloads/resnet-training-cpu': (197.44802653944336, 197.44802653944336),
                           'faas-workloads/resnet-inference-gpu': (0.12844885587692262, 0.7286638289081807),
                           'faas-workloads/mobilenet-inference-tpu': (0.5475386333465576, 0.5475386333465576)}

mean_execution_times = {
    'faas-workloads/tf-gpu': {'nx': 1.1731165671348571, 'nano': 0.6205120253562927, 'xeongpu': 0.36531192779541016,
                              'tx2': 1.8881448245048522},
    'faas-workloads/speech-inference-gpu': {'nx': 1.6482071661949158, 'nano': 4.544314980506897,
                                            'xeongpu': 0.747898964881897, 'tx2': 3.3071153211593627},
    'faas-workloads/resnet-training-gpu': {'nx': 141.99886816077762, 'nano': 847.171815255109,
                                           'xeongpu': 32.12711091739376, 'tx2': 228.1172546479437},
    'faas-workloads/speech-inference-tflite': {'nx': 2.6837231802940367, 'nano': 3.8856320667266844,
                                               'xeoncpu': 1.0779976797103883, 'xeongpu': 1.0779976797103883,
                                               'tx2': 3.4396099495887755, 'rpi3': 16.559413411617278,
                                               'rpi4': 6.7493284320831295, 'nuc': 1.0626225304603576,
                                               'coral': 7.2140442943573, 'rockpi': 3.815823349952698},
    'faas-workloads/resnet-preprocessing': {'nx': 6.081014342308045, 'nano': 7.95175134897232,
                                            'xeoncpu': 2.6644398760795593, 'xeongpu': 2.6644398760795593,
                                            'tx2': 6.387359216213226, 'rpi3': 30.478582429331404,
                                            'rpi4': 19.502971405982972, 'nuc': 2.5259750509262084,
                                            'coral': 10.51350971698761, 'rockpi': 7.646788821220398},
    'faas-workloads/mobilenet-inference-tflite': {'nx': 0.33246784210205077, 'nano': 0.4461205530166626,
                                                  'xeoncpu': 0.2831557750701904, 'xeongpu': 0.2831557750701904,
                                                  'tx2': 0.3282004809379578, 'rpi3': 2.0728497886657715,
                                                  'rpi4': 1.2829871106147765, 'nuc': 0.2803403902053833,
                                                  'coral': 0.6648596024513245, 'rockpi': 0.5154055190086365},
    'faas-workloads/python-pi': {'nx': 0.8302664136886597, 'nano': 0.7489693427085876, 'xeoncpu': 71.5935873889923,
                                 'xeongpu': 71.5935873889923, 'tx2': 0.5480701279640198, 'rpi3': 25.670029578208922,
                                 'rpi4': 23.58836589574814, 'nuc': 0.24590346574783326, 'coral': 1.8222162294387818,
                                 'rockpi': 0.9157870554924011},
    'faas-workloads/fio': {'nx': 13.771749126911164, 'nano': 19.639060771465303, 'xeoncpu': 1.143465440273285,
                           'xeongpu': 1.143465440273285, 'tx2': 4.314488160610199, 'rpi3': 28.66214724727299,
                           'rpi4': 27.808915467262267, 'nuc': 1.1227234315872192, 'rockpi': 21.987778232097625},
    'faas-workloads/resnet-inference-cpu': {'nx': 0.5094686794281006, 'nano': 0.9278767776489257,
                                            'xeoncpu': 0.1696533441543579, 'xeongpu': 0.1696533441543579,
                                            'tx2': 0.7448488783836364, 'rpi3': 5.095058534145355,
                                            'rpi4': 2.9077649402618406, 'nuc': 0.15748303413391113,
                                            'coral': 2.1848344469070433, 'rockpi': 1.3783342552185058},
    'faas-workloads/resnet-training-cpu': {'nuc': 197.44802653944336},
    'faas-workloads/resnet-inference-gpu': {'nx': 0.38789546728134155, 'nano': 0.7286638289081807,
                                            'xeongpu': 0.12844885587692262, 'tx2': 0.39461688995361327},
    'faas-workloads/mobilenet-inference-tpu': {'coral': 0.5475386333465576}}

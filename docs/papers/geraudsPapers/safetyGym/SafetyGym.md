# SafetyGym



---

## Page 1

Safety-Gymnasium: A Unified Safe Reinforcement
Learning Benchmark
JiamingJi1,∗,BorongZhang1,∗,JiayiZhou1,∗,XuehaiPan1,WeidongHuang1
RuiyangSun1,YiranGeng1,YifanZhong1,2,JuntaoDai1,YaodongYang1,†
1 InstituteforAI,PekingUniversity
2 BeijingInstituteforGeneralArtificialIntelligence(BIGAI)
{jiamg.ji, borongzh}@gmail.com, gaiejj@outlook.com
yaodong.yang@pku.edu.cn
Abstract
Artificialintelligence(AI)systemspossesssignificantpotentialtodrivesocietal
progress. However, their deployment often faces obstacles due to substantial
safetyconcerns. Safereinforcementlearning(SafeRL)emergesasasolutionto
optimizepolicieswhilesimultaneouslyadheringtomultipleconstraints,thereby
addressingthechallengeofintegratingreinforcementlearninginsafety-criticalsce-
narios. Inthispaper,wepresentanenvironmentsuitecalledSafety-Gymnasium,
which encompasses safety-critical tasks in both single and multi-agent scenar-
ios, accepting vector and vision-only input. Additionally, we offer a library of
algorithms named Safe Policy Optimization (SafePO), comprising 16 state-of-
the-artSafeRLalgorithms. Thiscomprehensivelibrarycanserveasavalidation
tool for the research community. By introducing this benchmark, we aim to
facilitate the evaluation and comparison of safety performance, thus fostering
thedevelopmentofreinforcementlearningforsafer, morereliable, andrespon-
sible real-world applications. The website of this project can be accessed at
https://sites.google.com/view/safety-gymnasium.
1 Introduction
AI systems possess enormous potential to spur societal progress. However, their deployment is
frequentlyhinderedbysubstantialsafetyconsiderations[1;2;3;4]. Distinctfrompurereinforcement
learning(RL),Safereinforcementlearning(SafeRL)seekstooptimizepolicieswhileconcurrently
adheringtomultipleconstraints,addressingthechallengeofemployingRLinscenarioswithcritical
safetyimplications[5;6;7;8;9].Thisstrategyprovesparticularlypertinentinreal-worldapplications
suchasautonomousvehicles[10]andhealthcare[11],wheresystemfailuresorunsafeactionscan
resultingraveconsequences,suchasaccidentsorharmtoindividuals. Inlargelanguagemodels
(LLMs),somestudieshavealsoshownthatthetoxicityofthemodelscanbereducedthroughSafeRL
[12;13]. Incorporatingsafetyconstraintsensuresadherencetopredefinedboundariesandregulatory
standards,fosteringtrustandenablingexplorationinenvironmentswithhigh-riskpotential. Overall,
SafeRLisinstrumentalinguaranteeingthedependableoperationofintelligentsystemsinintricate
andhigh-stakedomains.
∗EqualContribution.†Correspondingauthor.
WorkdonewhenJiayiZhouvisitedPekingUniversity.
37thConferenceonNeuralInformationProcessingSystems(NeurIPS2023)TrackonDatasetsandBenchmarks.


---

## Page 2

SimulationenvironmentshavebecomeinstrumentalinfosteringtheadvancementofRL.Eminent
examples such as Gym [14], Atari [15], and dm-control [16] underline their importance. These
versatile platforms permit researchers to swiftly design and execute varied tasks, thus enabling
efficient evaluation of algorithmic effectiveness and intrinsic limitations. However, within the
sphereofSafeRL,thereisanotabledearthofdedicatedsimulationenvironments,whichimpedes
comprehensiveexplorationofSafeRL.Inrecentyears,therehavebeenstridestoaddressthisgap.
DeepMindpresentedAI-Safety-Gridworlds,asuiteofRLenvironmentsshowcasingvarioussafety
propertiesofintelligentagents[17]. Afterward,OpenAIintroducedtheSafetyGymbenchmarksuite,
acollectionofhigh-dimensionalcontinuouscontrolenvironmentsincorporatingsafety-robottasks
[18]. Overthepasttwoyears,severaladditionalenvironmentshavebeendevelopedbyresearchers,
includingsafe-control-gym[19],MetaDrive[20],etc.
ComparedtoSafetyGym1Safety-Gymnasiuminheritsandexpandsthesettingsofsometasksof
SafetyGym,aimingtobolsterthecommunity’sgrowthfurther. ComparedwithSafetyGym,wehave
madethefollowingmajorimprovements:
• Refactoring of the physics engine. Safety Gym utilizes mujoco-py to enable Python-based
customizationofMuJoCocomponents. However,mujoco-pystoppedupdatesandsupportafter
2021. In contrast, Safety-Gymnasium supports MuJoCo directly, eliminating the reliance
onmujoco-py. ThisfacilitatesaccesstothelatestMuJoCofeatures(e.g.,renderingspeedand
accuracyimproved,etc.)andlowerstheentrybarrier,particularlyduetomujoco-py’sdependency
onspecificGCCversionsandmore.
• ExtensionofAgentandTaskComponents. SafetyGyminitiallysupportsonlythreeagentsand
tasks. Onthisbasis,Safety-Gymnasiumhasbeenfurtherexpanded,introducingmorediverse
agentsandtaskcomponentsandexpandingsafetytaskstocovermulti-agentdomains. Finally,
Safety-Gymnasium launched a high-dimensional test component based on Issac-Gym [21],
furtherenrichingthebenchmark.
• EnhancedVisualTaskSupport. ThevisualcomponentsofSafetyGymaresimplistic(consist-
ingofbasicgeometricshapes),andmujoco-pyreliesonOpenGLforvisualrendering,which
resultsinsignificantvirtualizationperformancelossonheadlessservers. Incontrast, Safety-
Gymnasium,builtonMuJoCo,achievesrenderingspeedsonCPUthataretwiceasfastasthe
former. Additionally,itoffersmorecomprehensivevisualcomponentsupport.
• EasyInstallationandHighCustomization. SafetyGymiscumbersometoinstallandrelies
heavilyontheunderlyingsoftware. OneofthedesignmotivationsofSafety-Gymnasiumisthe
easeofusesothateveryonecanfocusonalgorithmdesign. Safety-Gymnasiumcanbeeasily
installedwithonesimplecommandpipinstallsafety-gymnasium. Whilebenefitingfromthe
highlyintegratedframework,Safety-Gymnasiumonlyneeds100linesofcodetocustomize
therequiredenvironment.
Inthiswork,weintroduceSafety-Gymnasium,acollectionofenvironmentsspecificallyforSafeRL,
builtupontheGymnasium[14;22]andMuJoCo[23]. EnhancingtheextantSafetyGymframework
[18],weaddressvariousconcernsandexpandthetaskscopetoincludevision-onlyandmulti-agent
scenarios. Additionally,wereleasedSafePO,asingle-filestylealgorithmlibrarycontainingover16
state-of-the-artalgorithms. Collectively,ourcontributionsareenumeratedasfollows:
• Environmental Components. We provide various safety-oriented tasks under the umbrella
of Safety-Gymnasium. These tasks encompass single-agent, multi-agent, and vision-based
challenges,eachwithvaryingconstraints. Ourenvironmentsarecategorizedintotwoprimary
types:Gymnasium-based,featuringagentsofescalatingcomplexityforalgorithmverificationand
comparison,andIssac-Gym-based,incorporatingsophisticatedagentsthatharnesstheparallel
processingpowerofIssac-gym’sGPU.ThisempowersresearcherstoexploreSafeRLalgorithms
incomplexscenarios. FurtherdetailscanbefoundinSection4.
• AlgorithmComponents. WeoffertheSafePOalgorithmlibrary,whichcomprisesasingle-file
stylehousing16diversealgorithms. Thesealgorithmsencompassbothsingle-agentandmulti-
agentapproaches,alongwithfirst-orderandsecond-ordervariants,aswellasLagrangian-based
1Again, we have no intention of attacking Safety Gym; the contribution of Safety Gym to the SafeRL
communitycannotbeignored, andSafetyGymalsoinspiredthiswork. Wehopethatthroughourefforts,
Safety-GymnasiumcanfurtherpromotethedevelopmentofSafeRLandgivebacktotheentireRLcommunity.
2


---

## Page 3

andProjection-basedmethods. Throughmeticulousdecoupling,eachalgorithm’scoderesidesin
anindividualfile. Amorein-depthexplorationofSafePOispresentedinSection5.
• InsightsandAnalysis. CombiningSafety-GymnasiumandSafePO,weconductadetailed
analysisofexistingalgorithms. Ouranalysisencompasses16algorithmsacross54distinctenvi-
ronments,coveringvariousscenariossuchassingle-agentandmulti-agentsetupswithvarying
constraintcomplexities. Thisanalysisdelvesintoeachalgorithm’sstrengths,constraints,and
avenuesforenhancement. Weprovideaccesstoallmetadata,fosteringcommunityverification
andencouragingfurtherresearch. FurtherdetailscanbefoundinSection6.
2 RelatedWork
SafetyEnvironments InRL,agentsneedtoexploreenvironmentstolearnoptimalpoliciesby
trial and error. It is currently typical to train RL agents mostly or entirely in simulation, where
safetyconcernsareminimal. However,weanticipatethatchallengesinsimulatingthecomplexities
of the real world (e.g., human-AI collaborative control [1; 2]) will cause a shift towards training
RLagentsdirectlyintherealworld, wheresafetyconcernsareparamount[20;24;25]. OpenAI
includessafetyrequirementsintheSafetyGym[18],whichisasuiteofhigh-dimensionalcontinuous
controlenvironmentsformeasuringresearchprogressonSafeRL.Safe-control-gym[19]allowsfor
constraintspecificationanddisturbanceinjectionontoarobot’sinputs,states,andinertialproperties
through a portable configuration system. DeepMind also presents a suite of RL environments,
AI-Safety-Gridworlds[17],illustratingvarioussafetypropertiesofintelligentagents.
SafeRLAlgorithms CMDPshavebeenextensivelystudiedfordifferentconstraintcriteria[26;
27; 28; 29]. With the rise of deep learning, CMDPs are also moving to more high-dimensional
continuouscontrolproblems. CPO[30]proposesthefirstgeneral-purposepolicysearchalgorithm
forSafeRLwithguaranteesfornear-constraintsatisfactionateachiteration. However,CPO’spolicy
updateshingeonTaylorapproximationsandtheinversionofhigh-dimensionalFisherinformation
matrices. Theseapproximationscanoccasionallyleadtoinappropriatepolicyupdates. FOCOPS[31]
appliesaprimal-dualapproachtosolvetheconstrainedtrustregionproblemdirectlyandsubsequently
projectsthesolutionbackintotheparametricpolicyspace. Similarly,CUP[32]offersnon-convex
implementationsthroughafirst-orderoptimizer,therebynotrequiringastrongapproximationofthe
convexityoftheobjective.
3 Preliminaries
3.1 ConstrainedMarkovdecisionprocess
SafeRL[6;33]isoftenformulatedasaConstrainedMarkovdecisionprocess(CMDP)[6],whichisa
tupleM=(S,A,P,R,C,µ,γ). HereS andAarethestatespaceandactionspacecorrespondingly.
P(s′|s,a)istheprobabilityofstatetransitionfromstos′ aftertakingactiona. R(s′|s,a)denotes
therewardobtainedbytheagentperformingactionainstatesandtransitioningtostates′. The
set C = (cid:8) (c ,b ) (cid:9)m , where c are cost functions: c : S ×A → R and the cost thresholds are
i i i=1 i i
b ,i=1,··· ,m. µ(·):S →[0,1]istheinitialstatedistributionandthediscountfactorγ ∈[0,1).
i
Astationaryparameterizedpolicyπ isaprobabilitydistributiondefinedonS×A,π (a|s)denotes
θ θ
the probability of taking action a in state s. We use Π = {π : θ ∈ Rp} to denote the set of
θ θ
all stationary policies and θ is the network parameter needed to be learned. Let P ∈ R|S|×|S|
denotes a state transition probability matrix and the components are: P [s,s′] =
πθP
(s′|s) =
(cid:80) π (a|s)P(s′|s,a),whichdenotesone-stepstatetransitionprobability π f θ romstos′by πθ executing
a∈A θ
π of θ . th F e in M al a ly rk , o w v e c l h et ai d n s π 0 θ st ( a s r ) ti = ng ( a 1 t − s γ i ) n (cid:80) du ∞ t c = e 0 d γ b t y P π p θ o ( li s c t y = π s| a s n 0 d )t d o µ b ( e s t ) he = st E ationary [d st µ at ( e s d ) i ] s t t o rib b u e ti t o h n e
0 θ πθ s0∼µ(·) πθ
discountedstatevisitationdistributiononinitialdistributionµ.
The objective function is defined via the infinite horizon discounted reward function where
for a given π , we have JR(π ) = E[ (cid:80)∞ γtR(s |s ,a )|s ∼µ,a ∼π ]. The cost func-
θ θ t=0 t+1 t t 0 t θ
tion is similarly specified via the following infinite horizon discount cost function: JC(π ) =
E[ (cid:80)∞ γtC (s |s ,a )|s ∼µ,a ∼π ]. i θ
t=0 i t+1 t t 0 t θ
3


---

## Page 4

Then,wedefinethefeasiblepolicysetΠ as: Π =∩m {π ∈Π and JC(π )≤b }. Thegoal
C C i=1 θ θ i θ i
ofCMDPistosearchtheoptimalpolicyπ : π =argmax JR(π ).
⋆ ⋆ πθ∈ΠC θ
3.2 ConstrainedMarkovGame
Safe multi-agent reinforcement learning is often formulated as a Constrained Markov Game
(N,S,A,P,µ,γ,R,C,b). Here,N ={1,...,n}isthesetofagents,S andA= (cid:81)n Aiarethe
i=1
statespaceandthejointactionspace(i.e.,theproductoftheagents’actionspaces),P:S×A×S →R
istheprobabilistictransitionfunction,µistheinitialstatedistribution,γ ∈[0,1)isthediscountfac-
tor,R:S×A→Risthejointrewardfunction,C = (cid:8) Ci(cid:9)i∈N isthesetofsetsofcostfunctions
j 1≤j≤mi
(everyagentihasmicostfunctions)oftheformCi :S×Ai →R,andfinallythesetofcorrespond-
j
ingcostthresholdisgivenbyb=
(cid:8) bi(cid:9)i∈N
. Attimestept,theagentsareinastates ,andevery
j 1≤j≤mi t
agentitakesanactionai accordingtoitspolicyπi(cid:0) ai |s (cid:1) . Togetherwithotheragents’actions,it
t t
givesajointactiona = (cid:0) a1,...,an(cid:1) andthejointpolicyπ(a|s)= (cid:81)n πi(cid:0) ai |s (cid:1) . Theagents
t t t i=1
receivetherewardR(s ,a ),meanwhileeachagentipaysthecostsCi(cid:0) s ,ai(cid:1) ,∀j = 1,...,mi.
t t j t t
Theenvironmentthentransitstoanewstates ∼P(·|s ,a ).
t+1 t t
TheobjectiveofrewardfunctionareJ(π) ≜ E [ (cid:80)∞ γtR(s ,a )],andcosts
functionareJi(π)≜E (cid:2)(cid:80)∞ s0∼ρ γ 0 t ,a C 0: i ∞(cid:0) ∼ s π , ,s a 1 i :∞(cid:1)(cid:3) ∼ ≤ p ci, t=0 ∀j = t 1, t ...,mi.
j s0∼ρ0,a0:∞∼π,s1:∞∼p t=0 j t t j
We are examining a fully cooperative setting where all agents share a common reward function.
Consequently,thegoalofsafemulti-agentRListoidentifytheoptimalpolicythatmaximizesthe
expectedtotalrewardwhilesimultaneouslyensuringthatthesafetyconstraintsofeachagentare
satisfied. Thenwedefinethefeasiblejointpolicysetπ =∩n {π ∈Π and Ji(π)≤ci,∀j =
C i=1 θ θ j j
1,...,mi}. ThegoalofCMGistosearchtheoptimalpolicyπ =argmax J(π ).
⋆ πθ∈ΠC θ
4 SafetyEnvironments: Safety-Gymnasium
Safety-Gymnasiumprovidesaseamlessinstallationprocessandminimalisticcodesnippetstobasic
examples,asshowninFigure1. Duetothelimitedspaceofthepaper,weprovideamoredetailed
description(e.g.,detailedinstructions,thecompositionoftherobot’sobservationspaceandaction
space,dynamicstructure,physicalparameters,etc.) inAppendixBandOnlineDocumentation2.
"""
Install from PyPI:
pip install safety-gymnasium
"""
import safety_gymnasium
# Create the safety-task environment
env = safety_gymnasium.make(“SafetyPointGoal1-v0”, render_mode=“human”)
# Reset the environment
obs, info = env.reset()
while True:
# Sample a random action
act = env.action_space.sample()
# Step the environment: costs are returned
obs, reward, cost, terminated, truncated, info = env.step(act)
if terminated or truncated:
break
Figure1: UsingSafety-Gymnasiumtocreate,step,renderaspecificsafety-taskenvironment.
4.1 Gynasium-basedLearningEnvironments
Inthissection,weintroduceGymnasium-basedenvironmentcomponentsfromthreeaspects: (1)the
robots(bothsingle-agentandmulti-agent);(2)thetasksthataresupportedwithintheenvironment;
(3)thesafetyconstraintsthatareupheld.
2OnlineDocumentation:www.safety-gymnasium.com
4


---

## Page 5

SupportedRobots AsshowninFigure2,Safety-Gymnasiuminheritsthreepre-existingagents
from Safety Gym [18], namely Point, Car, and Doggo. By meticulously adjusting the model
parameters,wehavesuccessfullymitigatedtheissueofexcessiveoscillationsduringtheruntimeof
PointandCaragents. Buildinguponthisfoundation,wehaveintroducedtwoadditionalrobots:
racecar[34;35],andant[23],toenrichthesingle-agentscenarios. Asformulti-agentrobots,we
haveleveragedcertainconfigurationsfrommulti-agentMuJoCo[36],deconstructingtheoriginal
single-agentstructureandenablingmultipleagentstocontroldistinctbodysegments. Thisdesign
choicehasbeenwidelyadoptedinvariousresearchworks[37;38;39].
(a).Point (b).Car (c).Racecar (d).Doggo (e).Ant
(a).Ant (b).HalfCheetah (c).Humanoid (d).Hopper (e).Walker2D
Figure 2: Upper: The Single-Agent Robots of Gymnasium-based Environments. Lower: The
Multi-AgentRobotsofGymnasium-basedEnvironments.
SupportedTasks AsshowninFigure3,theGymnasium-basedlearningenvironmentssupportthe
followingtasks. Foramoredetailedtaskspecification,pleaserefertoouronlinedocumentation3.
• Velocity. Therobotaimstofacilitatecoordinatedlegmovementoftherobotintheforward(right)
directionbyexertingtorquesonthehinges.
• Run. Therobotstartswitharandominitialdirectionandaspecificinitialspeedasitembarkson
ajourneytoreachtheoppositesideofthemap.
• Circle. Therewardismaximizedbymovingalongthegreencircleandnotallowedtoenterthe
outsideoftheredregion,soitsoptimalpathfollowsthelinesegmentsADandBC.
• Goal. Therobotnavigatestomultiplegoalpositions. Aftersuccessfullyreachingagoal, its
locationisrandomlyresetwhilemaintainingtheoveralllayout.
• Push. Theobjectiveistomoveaboxtoaseriesofgoalpositions. Likethegoaltask,anew
randomgoallocationisgeneratedaftereachachievement.
• Button.Theobjectiveistoactivateaseriesofgoalbuttonsdistributedthroughouttheenvironment.
Theagent’sgoalistonavigatetowardsandcontactthecurrentlyhighlightedbutton,knownas
thegoalbutton.
SupportedConstraints AsshowninFigure3,theGymnasium-basedenvironmentssupportthe
followingconstraints. Foramoredetailedtaskspecification,pleaserefertoouronlinedocumentation.
• Velocity-ConstraintinvolvessafetytasksusingMuJoCoagents[23]. Inthesetasks,agentsaim
forhigherrewardbymovingfaster,buttheymustalsoadheretovelocityconstraintsforsafety.
Specifically, in a two-dimensional plane, the cost is computed as the Euclidean norm of the
agent’svelocities(v andv ).
x y
3TaskSpecificationDocumentation:https://www.safety-gymnasium.com/en/latest/components_
of_environments/tasks.html
5


---

## Page 6

(a).Velocity (b).Run (c).Circle (d).Goal (e).Button (f).Push
(a).VelocityConstraints (b).Pillars (c).Hazards (d).Sigwalls (e).Vases (f).Gremlins
Figure3: Upper: TasksofGymnasium-basedEnvironments;Lower: ConstraintsofGymnasium-
basedEnvironments.
• Pillars are employed to represent large cylindrical obstacles within the environment. In the
generalsetting,contactwithapillarincurscosts.
• Hazardsareutilizedtomodelareaswithintheenvironmentthatposearisk,resultingincosts
whenanagententerssuchareas.
• SigwallsaredesignedspecificallyforCircletasks. Crossingthewallfrominsidethesafeareato
theoutsideincurscosts.
• Vasesrepresentstaticandfragileobjectswithintheenvironment. Touchingordisplacingthese
objectsincurscostsfortheagent.
• Gremlinsrepresentmovingobjectswithintheenvironmentthatcaninteractwiththeagent.
4.1.1 Vision-onlytasks
Vision-onlySafeRLhasgainedsignificantattentionasafocalpointofresearch,primarilydueto
itsapplicabilityinreal-worldcontexts[40;41]. WhiletheinitialiterationofSafetyGymoffered
rudimentaryvisualinputsupport,thereisroomforenhancingtherealismofitsenvironment. To
effectivelyevaluatevision-basedSafeRLalgorithms,wehavedevisedamorerealisticvisualenvi-
ronmentutilizingMuJoCo. ThisenhancedenvironmentfacilitatestheincorporationofbothRGB
andRGB-Dinputs(asshowninFigure5). AnexemplarofthisenvironmentisdepictedinFigure4,
whilecomprehensivedescriptionsareavailableinAppendixB.5.
(a)Race (b)TheVisionInputofRace (c)FormulaOne (d)TheVisionInputofFormulaOne
Figure4: Vision-onlyTasksofGymnasium-basedEnvironments.
4.2 Issac-Gym-basedLearningEnvironments
Inthissection, weintroduceSafety-DexterousHands, acollectionofenvironmentsbuiltupon
DexterousHands [42] and the Isaac Gym engine [21]. Leveraging GPU capabilities, Safety-
DexterousHandsenableslarge-scaleparallelsamplecollection,significantlyacceleratingthetraining
process. Theenvironmentssupportbothsingle-agentandmulti-agentsettings. Theseenvironments
involvetworobotichands(refertoFigure6(a)and(b)). Ineachepisode,aballrandomlydescends
neartherighthand. Therighthandneedstograspandlaunchtheballtowardthelefthand,which
subsequentlycatchesanddepositsitatthetargetlocation.
6


---

## Page 7

BGR
D-BGR
(a)step=0 (b)step=150 (c)step=300 (d)step=450 (e)step=600 (f)step=750
Figure5: TheRGBandRGB-DinputofGymnasium-basedEnvironments.
(a) Hand Catch Over (b): Hand Over (c): Dynamics (d): Safety Joint (e): Safety Finger
Figure6: TasksofSafety-DexterousHands.
Fortimestept,letx ,x tobethepositionoftheballandthegoal,d todenotethepositional
b,t g,t p,t
distancebetweentheballandthegoald = ∥x −x ∥ . Letd denotetheangulardistance
p,t b,t g,t 2 a,t
between the object and the goal, and the rotational difference is d = 2arcsinmin{|d |,1.0}.
r,t a,t
Therewardisdefinedasfollows,r =exp{−0.2(αd +d )},whereαisaconstantbalanceof
t p,t r,t
positionalandrotationalreward.
SafetyJointconstrainsthefreedomofjoint④oftheforefinger(refertoFigure6(c)and(d)).Without
theconstraint,joint④hasfreedomof[−20°,20°].Thesafetytasksrestrictjoint④within[−10°,10°].
Letang_4betheangleofjoint④,andthecostisdefinedas: c =I(ang_4̸∈[−10°,10°]).
t
SafetyFingerconstrainsthefreedomofjoints②,③and④offorefinger(refertoFigure6(c)and
(e)). Withouttheconstraint,joints②and③havefreedomof[0°,90°]andjoint④of[−20°,20°].
The safety tasks restrict joints ②, ③, and ④ within [22.5°,67.5°], [22.5°,67.5°], and [−10°,10°]
respectively. Letang_2,ang_3,ang_4betheanglesofjoints②,③,④,andthecostisdefinedas:
c =I(ang_2̸∈[22.5°,67.5°], orang_3̸∈[22.5°,67.5°], orang_4̸∈[−10°,10°]). (1)
t
5 SafePolicyOptimizationAlgorithms: SafePO
ThissectionprovidesadetaileddiscussionofthedesignofSafePO.Featuressuchasstrongperfor-
mance,extensibility,customization,visualization,anddocumentationareallpresentedtodemonstrate
theadvantagesandcontributionsofSafePO.
Correctness Forabenchmark,itiscriticaltoensureitscorrectnessandreliability. Firstly,each
algorithmisimplementedstrictlyaccordingtotheoriginalpaper(e.g.,ensuringconsistencywiththe
gradientflowoftheoriginalpaper,etc.). Secondly,wecompareourimplementationwiththoseline
bylineforalgorithmswithacommonlyacknowledgedopen-sourcecodebasetodouble-checkthe
correctness. Finally,wecompareSafePOwithexistingbenchmarks(e.g.,Safety-Starter-Agents4 and
RL-Safety-Algorithms5)andSafePOoutperformsorachievescomparableperformancewithother
existingimplementations,asshowninTable1.
4Safety-Starter-Agents:https://github.com/openai/safety-starter-agents
5RL-Safety-Algorithms:https://github.com/SvenGronauer/RL-Safety-Algorithms
7


---

## Page 8

Safe Policy Constrained SafePO
Optimization
MAPPO-Lag
Safe Navigation: Button, Push, Multi-Goal, etc.
PPO-Lag
Lagrangian Pure Policy
RCPO
HAPPO
TRPO-Lag
CPPO-PID PIDControl MAPPO
Safe Manipulation: ShadowHandOver, FreightFrankaCloseDrawer, etc.
CUP PPO
FOCOPS Two Projection
Stage PG
PCPO
CPO One NaturalPG
Safe Velocity: Multi/Single-agent, Ant, Swimmer, Humanoid, etc. Stage
MACPO TRPO
Single / Multi-Agent Pipeline
Safety Vision: FormulaOne, Race, Building, Fading, etc.
Environment Wrapper Gymnasium / Isaac-Gym API
User Interface Keyboard Efficient Policy Customized Benchmarking
Controller Command Evaluator Configuration Tools
Figure7: TheArchitectureofSafePO
Extensibility SafePOenjoyshighextensibilitythankstoitsarchitecture(asshowninFigure7).
NewalgorithmscanbeintegratedintoSafePObyinheritingfrombasealgorithmsandonlyimple-
mentingtheiruniquefeatures. Forexample,weintegratePPObyinheritingfrompolicygradient
andonlyaddingtheclipratiovariableandrewritingthefunctionthatcomputesthelossofpolicyπ.
Similarly,algorithmscanbeeasilyaddedtoSafePO.
LoggingandVisualization AnothernecessaryfunctionalityofSafePOisloggingandvisualization.
SupportingbothTensorBoardandWandB,weoffercodeforvisualizingmorethan40parametersand
intermediatecomputationresultstoinspectthetrainingprocess.Standardparametersandmetricssuch
asKL-divergence,SPS(steppersecond),andcostvariancearevisualizeduniversally.Specialfeatures
ofalgorithmsarealsoreported, suchastheLagrangianmultiplierofLagrangian-basedmethods,
gTH−1g,gTH−1b,ν∗,andλ∗ of CPO, proportional, integral, and derivative of PID-Lagrangian
algorithms,etc. Duringtraining,userscaninspectthechangesofeveryparameter,collectthelogfile,
andobtainsavedcheckpointmodels. Thecompleteandcomprehensivevisualizationallowseasier
observation,modelselection,andcomparison.
Documentation Inadditiontoitscodeimplementation,SafePOcomeswithanextensivedocu-
mentation6. Weincludedetailedguidanceoninstallationandproposesolutionstocommonissues.
Moreover,weprovideinstructionsonsimpleusageandadvancedcustomizationofSafePO.Official
informationconcerningmaintenance,ethical,andresponsibleusearestatedclearlyforreference.
Table 1: A comparison between SafePO and other implementations. Results are based on 10
evaluationiterationsusingover3seedsundercost_limit=25.00.J¯Rstandsfornormalizedreward
fromPPO’sperformance,J¯C signifiesnormalizedcostrelativetocost_limit,andAvgR/AvgC
representstheratioofthemeansofbothacross10environments. The↑indicateshigherrewardsare
better,whilethe↓indicateslowercosts(whenbeyondthethresholdof1.00)arebetter. Grayand
Blackdepictsviolationandcompliancewiththecost_limit.
CPO TRPO-Lag PPO-Lag FOCOPS
SafePO(Ours) SafetyStarterAgents RL-Safety-Algorithms SafePO(Ours) SafetyStarterAgents RL-Safety-Algorithms SafePO(Ours) SafetyStarterAgents SafePO(Ours) OriginalImplementation
SafetyNavigation J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓
CARBUTTON1 0.08 1.75 0.34 3.65 -0.06 3.30 -0.04 1.08 0.02 0.78 -0.05 0.63 0.01 0.47 0.02 0.67 0.04 1.21 0.53 6.02
CARGOAL1 0.78 1.63 0.94 2.49 0.46 1.25 0.82 1.09 0.72 1.04 0.72 0.91 0.43 0.39 0.52 0.52 0.52 0.93 0.79 2.45
POINTBUTTON1 0.12 1.61 0.70 3.01 0.03 3.25 0.27 1.29 0.21 0.92 0.04 0.87 0.22 1.32 0.17 0.96 0.25 1.53 0.70 3.74
POINTGOAL1 0.78 1.10 0.81 1.99 0.28 2.05 0.72 0.91 0.65 0.94 0.33 0.72 0.47 1.50 0.66 0.77 0.56 1.32 0.81 1.53
SafetyVelocity J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓
ANTVEL 0.52 0.56 0.31 0.93 0.40 1.09 0.53 0.15 0.32 0.76 0.44 0.70 0.54 0.22 0.31 0.61 0.55 0.60 0.52 0.39
HALFCHEETAHVEL 0.40 0.23 0.30 1.13 0.31 0.97 0.43 1.01 0.25 0.79 0.43 0.67 0.44 0.04 0.30 0.93 0.42 0.12 0.44 0.04
HOPPERVEL 0.73 0.48 0.35 0.93 0.26 0.68 0.59 0.71 0.41 1.11 0.24 0.57 0.58 0.89 0.29 1.20 0.66 0.30 0.74 0.53
HUMANOIDVEL 0.71 0.01 0.05 0.19 0.36 0.83 0.72 2.38 0.05 0.01 0.71 0.79 0.72 0.76 0.07 0.09 0.71 0.93 0.73 0.43
SWIMMERVEL 0.51 0.82 0.38 1.11 0.41 0.82 0.66 0.84 0.43 1.67 0.41 1.02 0.57 1.11 0.38 1.18 0.47 1.30 0.68 0.71
WALKER2DVEL 0.39 0.81 0.44 1.85 0.05 0.67 0.51 0.77 0.46 0.67 0.51 1.34 0.44 0.20 0.47 0.81 0.50 0.68 0.48 0.74
AvgR/AvgC 0.56 0.27 0.17 0.51 0.40 0.46 0.64 0.41 0.52 0.39
6SafePO’sDocumentation:https://safe-policy-optimization.readthedocs.io
8


---

## Page 9

6 ExperimentsandAnalysis
(a) Average Episodic Reward of Algorithms (b) Bar Chart Categorizing Algorithms into Four Classes Based on Average Episodic Cost
Figure8: Abarchartanalyzingtheperformanceofdifferentalgorithms. Theleftgraphcompares
episodicrewardwithPPO-Lag[18](orMAPPO-Lag[39]formulti-agent). Therightgraphshows
episodiccostsproportionallyundervaryingconstraints. Single-agentdataisfrom40navigationand
6velocitytasks,andmulti-agentdataisfromall8velocitytasksinSafety-Gymnasium.
Table2: Theperformanceofsingle-agentalgorithms. J¯R standsfornormalizedrewardfromPPO’s
performance, and J¯C signifies normalized cost relative to cost_limit. The ↑ indicates higher
rewardsarebetter,whilethe↓indicateslowercosts(whenbeyondthethresholdof1.00)arebetter.
GrayandBlackdepictsbreachandcompliancewiththecost_limit,whileGreenrepresentsthe
optimalpolicy,maximizingrewardwithinsafetyconstraints.
PPO PPO-Lag TRPO-Lag CPPO-PID RCPO CPO PCPO CUP FOCOPS
SafetyNavigation J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓
ANTBUTTON1 1.00 4.42 0.09 0.86 0.23 1.95 0.10 0.70 0.16 2.07 0.12 4.01 0.03 1.01 0.03 0.17 0.01 0.46
ANTCIRCLE1 1.00 16.81 0.79 2.56 0.65 1.05 0.69 1.90 0.63 1.04 0.47 1.07 0.28 1.87 0.60 0.82 0.02 1.22
ANTGOAL1 1.00 1.81 0.26 0.94 0.25 0.74 0.47 1.94 0.29 0.78 0.19 0.55 0.09 0.42 0.34 1.33 0.09 0.67
ANTPUSH1 1.00 1.90 0.13 0.00 0.30 0.00 0.13 0.00 0.33 0.00 0.17 0.00 0.07 0.00 0.20 0.00 -0.30 0.03
CARBUTTON1 1.00 16.09 0.01 0.47 -0.04 1.08 -0.10 0.40 -0.19 1.73 0.08 1.75 0.02 1.90 0.04 5.50 0.04 1.21
CARCIRCLE1 1.00 8.42 0.81 0.82 1.69 2.77 1.61 1.79 1.70 3.11 1.67 3.13 1.41 1.99 0.76 1.04 0.84 1.12
CARGOAL1 1.00 2.38 0.43 0.39 0.82 1.09 0.03 2.47 0.55 0.86 0.78 1.63 0.61 1.42 0.19 0.63 0.52 0.93
CARPUSH1 1.00 7.16 0.46 0.78 1.38 0.70 0.03 0.47 1.11 1.42 0.83 1.14 0.64 2.36 0.32 0.95 0.29 0.36
DOGGOBUTTON1 1.00 7.57 0.01 0.03 0.00 1.27 0.01 0.07 0.01 0.09 0.00 0.15 0.00 0.25 0.02 0.45 0.06 3.68
DOGGOCIRCLE1 1.00 33.14 0.77 0.46 0.67 1.37 0.82 2.16 0.55 1.32 0.66 1.22 0.31 0.55 0.80 2.04 0.73 4.49
DOGGOGOAL1 1.00 2.28 0.05 0.00 0.18 0.69 0.00 0.00 0.16 2.08 0.30 0.50 0.00 0.00 0.00 0.90 0.04 1.27
DOGGOPUSH1 1.00 1.31 0.09 0.00 0.53 0.78 0.32 0.44 0.54 1.55 0.46 0.00 0.36 0.00 0.30 0.68 0.64 3.40
POINTBUTTON1 1.00 6.06 0.22 1.32 0.27 1.29 0.00 0.84 0.12 1.13 0.12 1.61 0.08 2.19 0.18 1.26 0.25 1.53
POINTCIRCLE1 1.00 8.10 0.86 0.93 1.67 1.35 1.72 2.09 1.66 1.42 1.69 1.74 1.33 2.26 0.82 0.62 0.84 0.89
POINTGOAL1 1.00 1.93 0.47 1.50 0.72 0.91 0.31 1.05 0.53 0.99 0.78 1.10 0.71 0.82 0.46 0.73 0.56 1.32
POINTPUSH1 1.00 2.31 0.98 1.33 0.85 1.00 0.35 0.35 5.30 0.94 2.22 0.80 1.72 1.25 2.32 0.80 1.13 2.51
RACECARBUTTON1 1.00 13.73 -0.01 1.94 -0.02 1.77 -0.16 2.06 -0.07 1.19 0.00 2.44 0.02 1.82 0.00 5.23 -0.10 3.37
RACECARCIRCLE1 1.00 15.87 0.83 1.90 0.80 2.18 0.58 1.33 0.83 2.07 0.79 0.81 0.22 2.87 0.74 3.53 0.77 2.11
RACECARGOAL1 1.00 4.26 0.26 0.51 1.19 0.77 -0.04 1.07 0.88 0.83 1.18 2.58 0.33 0.24 0.13 1.22 0.31 0.62
RACECARPUSH1 1.00 2.34 -0.40 0.00 0.74 1.79 -0.84 2.87 0.58 1.92 0.94 0.13 -0.16 0.18 -0.06 3.79 0.30 2.04
SafetyVelocity J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓
ANTVEL 1.00 38.33 0.54 0.22 0.53 0.15 0.51 0.41 0.52 0.56 0.52 0.56 0.38 0.41 0.55 0.94 0.55 0.60
HALFCHEETAHVEL 1.00 36.77 0.44 0.00 0.43 1.01 0.48 0.04 0.36 0.56 0.40 0.23 0.25 0.63 0.40 0.17 0.42 0.12
HOPPERVEL 1.00 22.00 0.58 0.89 0.59 0.71 0.73 0.44 0.58 0.59 0.73 0.48 0.65 0.51 0.73 0.21 0.66 0.30
HUMANOIDVEL 1.00 38.42 0.72 0.76 0.72 2.38 0.73 0.00 0.68 0.82 0.71 0.01 0.64 0.01 0.68 0.80 0.71 0.93
SWIMMERVEL 1.00 6.61 0.57 1.11 0.66 0.84 0.91 0.92 0.54 0.90 0.51 0.82 0.50 0.69 0.59 0.96 0.47 1.30
WALKER2DVEL 1.00 36.11 0.44 0.20 0.51 0.77 0.27 0.36 0.49 0.15 0.39 0.81 0.27 0.71 0.44 0.18 0.50 0.16
Reward and Cost. Episodic reward and cost exhibit a trade-off relationship. Unconstrained
algorithmsaimtomaximizerewardthroughriskybehaviors. HAPPO[37]achieveshigherrewards
comparedtoMAPPO[38]across8velocity-basedtasks,accompaniedbyasimultaneousincrease
inaveragecosts. SafeRLalgorithmstend to maximizereward whileadheringtoconstraints. As
depictedinTable2,inthevelocitytask,comparedtoPPO[43],PPO-Lag[18]achievesareduction
of98%incostwhileonlyexperiencingadecreaseof45%inreward.
RandomnessandOscillation. Therandomnessoftasksiscorrelatedwiththeoscillationofalgo-
rithms’performance. AllSafeRLalgorithmsachieveaverageepisodiccostswithinthecost_limit
for velocity tasks. The divergence in episodic rewards between algorithms is negligible, and the
distributionofoptimalpoliciesistightlyclustered. However,pronouncedoscillationsarepresent
innavigationtaskscharacterizedbyhighstochasticity. Outofthe20navigationtasksexamined,
9


---

## Page 10

optimalpoliciesarespreadoutmore,leadingtoobservabledifferencesinalgorithmperformance
acrossvarioustasks.
Lagrangian vs. Projection. In contrast
to projection-based methods, the Lagrangian- Table3: ThenormalizedperformanceofSafePO’s
basedmethodstendtodisplaymoreoscillation. multi-agentalgorithmsonSafety-Gymnasium.
Anotabledisparitybecomesapparentuponex-
aminingtheoscillatorypatternsintheepisodic MAPPO HAPPO MAPPO-Lag MACPO
costaroundthedesignatedsafetyconstraintsdur- SafetyVelocity J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓ J¯R↑ J¯C↓
2X4ANTVEL 1.00 35.76 1.26 39.12 0.57 0.00 0.51 0.14
ingtraining,aspresentedinFigure8(b). Both 4X2ANTVEL 1.00 38.01 1.07 34.34 0.50 0.00 0.50 0.01
CPO[30]andPPO-Lag[18]demonstrateoscil- 2 6 X X 3 1 H H A A L L F F C C H H E E E E T T A A H H V V E E L L 1 1 . . 0 0 0 0 3 3 9 9 . . 0 2 2 3 1 1 . . 1 0 1 9 3 3 7 7 . . 7 7 0 4 0 0 . . 3 2 5 8 0 0 . . 0 0 1 2 0 0 . . 4 3 9 6 1 0 . . 2 3 8 7
lations; however,thoseexhibitedbyPPO-Lag 3 9 X |8 1 H H U O M P A PE N R O V ID E V L EL 1 1 . . 0 0 0 0 2 6 2 . . 3 5 4 8 1 2 . . 0 7 4 9 2 1 2 7 . . 0 1 5 8 0 0 . . 4 5 7 4 0 0 . . 0 8 0 4 0 0 . . 2 5 2 3 1 1 . . 0 3 3 0
aremoreconspicuous. Thisdiscrepancyisman- 2X3WALKER2DVEL 1.00 22.99 1.55 33.67 0.60 0.01 0.27 1.21
ifestedinahigherproportionofinstancesclas-
sified as Strongly Unsafe and Strongly Safe for PPO-Lag, while CPO maintains a more centered
distribution. Nevertheless,anexcessivelycautiouspolicyhasthepotentialtoundermineperformance.
In contrast, the projection-based method PCPO [3] exhibits lower average costs and rewards in
navigationandvelocitytasksthanCPO.Thisdistinctionisfurtheraccentuatedwhenexaminingthe
contrastbetweenMACPOandMAPPO-Lag.
Lagrangian vs. PID-Lagrangian. Incorporating a PID controller within the Lagrangian-based
frameworkprovestobeeffectiveinmitigatinginherentoscillations.AsshowninFigure8,CPPO-PID
[44]displaysepisodicrewardsduringtrainingthatcloselyresemblethoseofPPO-Lag. However,
CPPO-PID demonstrates a reduced frequency of instances entering the Strongly Unsafe region,
resultinginamoresignificantproportionofSafestatesandimprovedsafetyperformance.
7 LimitationsandFutureWorks
Ensuringsafetyremainsaparamountconcern. Acrossvarioustasks,safetyconcernscanbetrans-
formedintocorrespondingconstraints.However,alimitationofthisstudyisitsinabilitytoencompass
all forms of constraints. For instance, safety constraints related to human-centric considerations
areparamountinhuman-AIcollaboration,yettheseconsiderationshavenotbeenfullyintegrated
withinthescopeofthisstudy. Thisworkfocusesonsafetytaskswithinasimulatedenvironmentand
introducesanextensivetestingcomponent. However,thetransferabilityoftheresultstocomplex
real-world safety-critical applications may be limited. A promising work for the future involves
transferringpolicyrefinedwithintheSafety-Gymnasiumtophysicalroboticplatforms,whichholds
profoundimplications.
10


---

## Page 11

References
[1] Tom Carlson and Yiannis Demiris. Increasing robotic wheelchair safety with collaborative
control: Evidencefromsecondarytaskexperiments. In2010IEEEInternationalConferenceon
RoboticsandAutomation,pages5582–5587.IEEE,2010.
[2] Zhu Ming Bi, Chaomin Luo, Zhonghua Miao, Bing Zhang, WJ Zhang, and Lihui Wang.
Safetyassurancemechanismsofcollaborativeroboticsystemsinmanufacturing. Roboticsand
Computer-IntegratedManufacturing,67:102022,2021.
[3] Tsung-YenYang,JustinianRosca,KarthikNarasimhan,andPeterJRamadge. Projection-based
constrainedpolicyoptimization. arXivpreprintarXiv:2010.03152,2020.
[4] FengshuoBai,HongmingZhang,TianyangTao,ZhihengWu,YannaWang,andBoXu. Picor:
Multi-task deep reinforcement learning with policy correction. Proceedings of the AAAI
ConferenceonArtificialIntelligence,37(6):6728–6736,Jun.2023.
[5] KeithWRossandRaviVaradarajan. Markovdecisionprocesseswithsamplepathconstraints:
thecommunicatingcase. OperationsResearch,37(5):780–790,1989.
[6] EitanAltman. ConstrainedMarkovdecisionprocesses. Routledge,2021.
[7] LinruiZhang,QinZhang,LiShen,BoYuan,XueqianWang,andDachengTao. Evaluating
model-freereinforcementlearningtowardsafety-criticaltasks. InProceedingsoftheAAAI
ConferenceonArtificialIntelligence,volume37(12),pages15313–15321,2023.
[8] JiamingJi,JiayiZhou,BorongZhang,JuntaoDai,XuehaiPan,RuiyangSun,WeidongHuang,
YiranGeng,MickelLiu,andYaodongYang. Omnisafe: Aninfrastructureforacceleratingsafe
reinforcementlearningresearch,2023.
[9] JiamingJi,TianyiQiu,BoyuanChen,BorongZhang,HantaoLou,KaileWang,YawenDuan,
ZhonghaoHe,JiayiZhou,ZhaoweiZhang,FanzhiZeng,KwanYeeNg,JuntaoDai,Xuehai
Pan,AidanO’Gara,YingshanLei,HuaXu,BrianTse,JieFu,StephenMcAleer,YaodongYang,
Yizhou Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. Ai alignment: A comprehensive
survey,2024.
[10] ShuoFeng,HaoweiSun,XintaoYan,HaojieZhu,ZhengxiaZou,ShengyinShen,andHenryX
Liu. Dense reinforcement learning for safety validation of autonomous vehicles. Nature,
615(7953):620–627,2023.
[11] YafeiOuandMahdiTavakoli. Towardssafeandefficientreinforcementlearningforsurgical
robotsusingreal-timehumansupervisionanddemonstration. In2023InternationalSymposium
onMedicalRobotics(ISMR),pages1–7.IEEE,2023.
[12] JiamingJi,MickelLiu,JuntaoDai,XuehaiPan,ChiZhang,CeBian,ChiZhang,RuiyangSun,
YizhouWang,andYaodongYang. Beavertails: Towardsimprovedsafetyalignmentofllmviaa
human-preferencedataset,2023.
[13] JosefDai,XuehaiPan,RuiyangSun,JiamingJi,XinboXu,MickelLiu,YizhouWang,and
YaodongYang. Saferlhf: Safereinforcementlearningfromhumanfeedback,2023.
[14] GregBrockman,VickiCheung,LudwigPettersson,JonasSchneider,JohnSchulman,JieTang,
andWojciechZaremba. Openaigym. arXivpreprintarXiv:1606.01540,2016.
[15] VolodymyrMnih,KorayKavukcuoglu,DavidSilver,AlexGraves,IoannisAntonoglou,Daan
Wierstra,andMartinRiedmiller. Playingatariwithdeepreinforcementlearning. arXivpreprint
arXiv:1312.5602,2013.
[16] SaranTunyasuvunakool,AlistairMuldal,YotamDoron,SiqiLiu,StevenBohez,JoshMerel,
TomErez,TimothyLillicrap,NicolasHeess,andYuvalTassa. dm_control: Softwareandtasks
forcontinuouscontrol. SoftwareImpacts,6:100022,2020.
[17] JanLeike,MiljanMartic,VictoriaKrakovna,PedroAOrtega,TomEveritt,AndrewLefrancq,
LaurentOrseau,andShaneLegg. Aisafetygridworlds. arXivpreprintarXiv:1711.09883,2017.
11


---

## Page 12

[18] AlexRay,JoshuaAchiam,andDarioAmodei. Benchmarkingsafeexplorationindeeprein-
forcementlearning. arXivpreprintarXiv:1910.01708,7(1):2,2019.
[19] ZhaocongYuan,AdamWHall,SiqiZhou,LukasBrunke,MelissaGreeff,JacopoPanerati,
and Angela P Schoellig. Safe-control-gym: A unified benchmark suite for safe learning-
basedcontrolandreinforcementlearninginrobotics. IEEERoboticsandAutomationLetters,
7(4):11142–11149,2022.
[20] Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, and Bolei Zhou.
Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning.
IEEEtransactionsonpatternanalysisandmachineintelligence,45(3):3461–3475,2022.
[21] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles
Macklin,DavidHoeller,NikitaRudin,ArthurAllshire,AnkurHanda,etal. Isaacgym: High
performancegpu-basedphysicssimulationforrobotlearning. arXivpreprintarXiv:2108.10470,
2021.
[22] FaramaFoundation. Astandardapiforsingle-agentreinforcementlearningenvironments,with
popularreferenceenvironmentsandrelatedutilities(formerlygym). https://github.com/F
arama-Foundation/Gymnasium,2022.
[23] EmanuelTodorov,TomErez,andYuvalTassa. Mujoco: Aphysicsengineformodel-based
control. In2012IEEE/RSJinternationalconferenceonintelligentrobotsandsystems,pages
5026–5033.IEEE,2012.
[24] Mengdi Xu, Zuxin Liu, Peide Huang, Wenhao Ding, Zhepeng Cen, Bo Li, and Ding Zhao.
Trustworthyreinforcementlearningagainstintrinsicvulnerabilities: Robustness,safety,and
generalizability. arXivpreprintarXiv:2209.08025,2022.
[25] ShangdingGu,LongYang,YaliDu,GuangChen,FlorianWalter,JunWang,YaodongYang,
andAloisKnoll. Areviewofsafereinforcementlearning: Methods,theoryandapplications.
arXivpreprintarXiv:2205.10330,2022.
[26] LodewijkCMKallenberg. Linearprogrammingandfinitemarkoviancontrolproblems. MC
Tracts,1983.
[27] JavierGarcıaandFernandoFernández. Acomprehensivesurveyonsafereinforcementlearning.
JournalofMachineLearningResearch,16(1):1437–1480,2015.
[28] JuntaoDai,JiamingJi,LongYang,QianZheng,andGangPan. Augmentedproximalpolicy
optimizationforsafereinforcementlearning. ProceedingsoftheAAAIConferenceonArtificial
Intelligence,37(6):7288–7295,Jun.2023.
[29] WeidongHuang,JiamingJi,BorongZhang,ChunheXia,andYaodongYang. Safedreamer:
Safereinforcementlearningwithworldmodels,2023.
[30] JoshuaAchiam,DavidHeld,AvivTamar,andPieterAbbeel. Constrainedpolicyoptimization.
InInternationalconferenceonmachinelearning,pages22–31.PMLR,2017.
[31] YimingZhang,QuanVuong,andKeithRoss. Firstorderconstrainedoptimizationinpolicy
space. AdvancesinNeuralInformationProcessingSystems,33:15338–15349,2020.
[32] LongYang,JiamingJi,JuntaoDai,YuZhang,PengfeiLi,andGangPan. Cup: Aconservative
updatepolicyalgorithmforsafereinforcementlearning,2022.
[33] RichardSSuttonandAndrewGBarto. Reinforcementlearning: Anintroduction. MITpress,
2018.
[34] PatrickLJacobs,StephenEOlvey,BradMJohnson,andKellyCohn. Physiologicalresponses
to high-speed, open-wheel racecar driving. Medicine and science in sports and exercise,
34(12):2085–2090,2002.
[35] JohannesBetz,AlexanderWischnewski,AlexanderHeilmeier,FelixNobis,TimStahl,Leonhard
Hermansdorfer,andMarkusLienkamp. Asoftwarearchitectureforanautonomousracecar. In
2019IEEE89thVehicularTechnologyConference(VTC2019-Spring),pages1–6.IEEE,2019.
12


---

## Page 13

[36] Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip Torr, Wendelin
Böhmer,andShimonWhiteson. Deepmulti-agentreinforcementlearningfordecentralized
continuouscooperativecontrol. arXivpreprintarXiv:2003.06709,19,2020.
[37] JakubGrudzienKuba,RuiqingChen,MuningWen,YingWen,FangleiSun,JunWang,and
YaodongYang. Trustregionpolicyoptimisationinmulti-agentreinforcementlearning. arXiv
preprintarXiv:2109.11251,2021.
[38] ChaoYu,AkashVelu,EugeneVinitsky,JiaxuanGao,YuWang,AlexandreBayen,andYiWu.
The surprising effectiveness of ppo in cooperative multi-agent games. Advances in Neural
InformationProcessingSystems,35:24611–24624,2022.
[39] ShangdingGu,JakubGrudzienKuba,MunningWen,RuiqingChen,ZiyanWang,ZhengTian,
JunWang,AloisKnoll,andYaodongYang. Multi-agentconstrainedpolicyoptimisation. arXiv
preprintarXiv:2110.02793,2021.
[40] YechengJasonMa,AndrewShen,OsbertBastani,andJayaramanDinesh. Conservativeand
adaptive penalty for model-based safe reinforcement learning. In Proceedings of the AAAI
conferenceonartificialintelligence,volume36(5),pages5404–5412,2022.
[41] YardenAs,IlnuraUsmanova,SebastianCuri,andAndreasKrause. Constrainedpolicyopti-
mizationviabayesianworldmodels. arXivpreprintarXiv:2201.09802,2022.
[42] Yuanpei Chen, Tianhao Wu, Shengjie Wang, Xidong Feng, Jiechuan Jiang, Zongqing Lu,
Stephen McAleer, Hao Dong, Song-Chun Zhu, and Yaodong Yang. Towards human-level
bimanualdexterousmanipulationwithreinforcementlearning. AdvancesinNeuralInformation
ProcessingSystems,35:5150–5163,2022.
[43] JohnSchulman,FilipWolski,PrafullaDhariwal,AlecRadford,andOlegKlimov. Proximal
policyoptimizationalgorithms. arXivpreprintarXiv:1707.06347,2017.
[44] AdamStooke,JoshuaAchiam,andPieterAbbeel. Responsivesafetyinreinforcementlearning
bypidlagrangianmethods.InInternationalConferenceonMachineLearning,pages9133–9143.
PMLR,2020.
[45] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
dimensional continuous control using generalized advantage estimation. arXiv preprint
arXiv:1506.02438,2015.
[46] DiederikPKingmaandJimmyBa. Adam:Amethodforstochasticoptimization. arXivpreprint
arXiv:1412.6980,2014.
13


---

## Page 14

A DetailsofExperimentalResults
A.1 HyperparametersAnalysis
ThissectionpresentsthedisclosureofSafePOhyperparameterssettingsandtheirrationales. We
employedtheGeneralizedAdvantageEstimation(GAE)[45]methodtoestimatethevaluesofrewards
andcostadvantagesandusedAdam[46]forlearningtheneuralnetworkparameters.
Single-agentAlgorithmSettings. Themodelsemployedinthesingle-agentalgorithmswere3-layer
MLPswithTanhactivationfunctionsandhiddenlayersizesof[64,64],formoreintricatenavigation
agentsAntandDoggo,hiddenlayersof[256,256]wereemployed.
Multi-agentAlgorithmsSettings. Themodelsemployedinthemulti-agentalgorithmswere3-layer
MLPswithReLUactivationfunctionsandhiddenlayersizesof[128,128].
Table4: HyperparametersofSafePOalgorithmsinSafety-Gymnasiumtasks. Second-orderalgo-
rithmssettheparameterstotheactormodeldirectly,insteadofiterativegradientdescent,sotheActor
LearningRateofthemaremarkedGray.
PG/PPO/PPO-Lag Value TRPO/TRPO-Lag Value CPPO-PID Value NPG/RCPO Value HAPPO/MAPPO/MAPPO-Lag Value
DiscountFactorγ 0.99 DiscountFactorγ 0.99 DiscountFactorγ 0.99 DiscountFactorγ 0.99 DiscountFactorγ 0.99
TargetKL 0.02 TargetKL 0.01 TargetKL 0.02 TargetKL 0.01 TargetKL 0.016
GAEλ 0.95 GAEλ 0.95 GAEλ 0.95 GAEλ 0.95 GAEλ 0.95
NumberofSGDIterations 40 NumberofSGDIterations 10 NumberofSGDIterations 40 NumberofSGDIterations 10 NumberofSGDIterations 5
TrainingBatchSize 20000 TrainingBatchSize 20000 TrainingBatchSize 20000 TrainingBatchSize 20000 TrainingBatchSize 10000
ActorLearningRate 0.0003 ActorLearningRate None ActorLearningRate 0.0003 ActorLearningRate None ActorLearningRate 0.0005
CriticLearningRate 0.0003 CriticLearningRate 0.001 CriticLearningRate 0.0003 CriticLearningRate 0.001 CriticLearningRate 0.0005
CostLimit 25.00 CostLimit 25.00 CostLimit 25.00 CostLimit 25.00 CostLimit 25.00
ClipCoefficient 0.20 ConjugateGradientIterations 15 ClipCoefficient 0.20 ConjugateGradientIterations 15 ClipCoefficient 0.20
LagrangianInitialValue 0.001 LagrangianInitialValue 0.001 PIDControllerKp 0.10 LagrangianInitialValue 0.001 LagrangianInitialValue 0.00001
LagrangianLearningRate 0.035 LagrangianLearningRate 0.035 PIDControllerKi 0.01 LagrangianLearningRate 0.035 LagrangianLearningRate 0.78
LagrangianOptimizer Adam LagrangianOptimizer Adam PIDControllerKd 0.01 LagrangianOptimizer Adam LagrangianOptimizer SGD
CPO Value PCPO Value CUP Value FOCOPS Value MACPO Value
DiscountFactorγ 0.99 DiscountFactorγ 0.99 DiscountFactorγ 0.99 DiscountFactorγ 0.99 DiscountFactorγ 0.99
TargetKL 0.01 TargetKL 0.01 TargetKL 0.02 TargetKL 0.02 TargetKL 0.01
GAEλ 0.95 GAEλ 0.95 GAEλ 0.95 GAEλ 0.95 GAEλ 0.95
NumberofSGDIterations 10 NumberofSGDIterations 10 NumberofSGDIteration 40 NumberofSGDIteration 40 NumberofSGDIteration 15
TrainingBatchSize 20000 TrainingBatchSize 20000 TrainingBatchSize 20000 TrainingBatchSize 20000 TrainingBatchSize 10000
ActorLearningRate None ActorLearningRate None ActorLearningRate 0.0003 ActorLearningRate 0.0003 ActorLearningRate None
CriticLearningRate 0.001 CriticLearningRate 0.001 CriticLearningRate 0.0003 CriticLearningRate 0.0003 CriticLearningRate 0.0005
CostLimit 25.00 CostLimit 25.00 CostLimit 25.00 CostLimit 25.00 CostLimit 25.00
ConjugateGradientIterations 15 ConjugateGradientIterations 15 ClipCoefficient 0.20 ClipCoefficient 0.20 ConjugateGradientIterations 10
CPOSearchingSteps 15 PCPOSearchingSteps 200 CUPλ 0.95 FOCOPSλ 1.50 MACPOSearchingSteps 10
StepFraction 0.80 StepFraction 0.80 CUPν 2.00 FOCOPSν 2.00 StepFraction 0.50
LagrangianMultiplierSettings. Lagrangian-basedmethodsaresensitivetohyperparameters. We
presentthefollowingdetaileddescriptionofthesettingsforboththenaiveandthePID-controlled
Lagrangianmultiplier.
• Lagrangian Initial Value: The initial value of the Lagrangian multiplier. It impacts the
early-stageperformanceoftheLagrangian-basedmethods. Ahigherinitialvaluepromotes
saferexplorationbutmayimpedetaskcompletion. Conversely,alowerinitialvaluedelays
theagent’sexplorationofsafepolicies.
• LagrangianLearningRate: ThelearningrateoftheLagrangianmultiplier. Ahighlearning
rateinducesexcessiveoscillations,impedesconvergencespeed,andhindersthealgorithm’s
abilitytoattainthedesiredsolution.Conversely,alowlearningrateslowsdownconvergence
andadverselyaffectstraining.
• PIDControllerKp: ThePIDcontroller’sproportionalgaindeterminestheoutput’sresponse
tochangesintheepisodiccosts. Ifpid_kpistoolarge,theLagrangianmultiplieroscillates,
andperformancedeteriorates. Ifpid_kpistoosmall,theLagrangianmultiplierupdates
slowly,alsoimpactingperformancenegatively.
• PIDControllerKd: ThePIDcontroller’sderivativegaingovernstheoutput’sresponseto
changesintheepisodiccosts. Ifpid_kdistoolarge,theLagrangianmultiplierbecomes
excessively sensitive to noise or changes in the episodic costs, leading to instability or
oscillations. Ifpid_kdistoosmall,theLagrangianmultipliermaynotrespondquicklyor
accuratelyenoughtochangesintheepisodiccosts.
• PIDControllerKi: ThePIDcontroller’sintegralgaindeterminesthecontroller’sabilityto
eliminatethesteady-stateerrorbyintegratingtheepisodiccostsovertime. Ifpid_kiistoo
large,theLagrangianmultipliermaybecomeoverlyresponsivetopreviouserrors,adversely
affectingperformance.
14


---

## Page 15

A.2 PerformanceTableofSafety-Gymnasium
Table5:TheperformanceofSafePOalgorithmsonSafety-Gymnasium.Allexperimentaloutcomes
were derived from 10 assessment iterations encompassing multiple random seeds and under the
experimentalsettingofcost_limit=25.00. The↑indicateshigherrewardsarebetter,whilethe
↓ indicates lower costs (when beyond the threshold of25.00) are better. Gray and Black depicts
breachandcompliancewiththecost_limit,whileGreenrepresentstheoptimalpolicy,maximizing
rewardwithinsafetyconstraints.
(a)TheperformanceofSafePOsingle-agentalgorithmsonSafety-Gymnasium.
PPO PPO-Lag CPPO-PID TRPO-Lag RCPO CPO PCPO CUP FOCOPS
SafetyNavigation JR JC JR JC JR JC JR JC JR JC JR JC JR JC JR JC JR JC
ANTBUTTON1 38.70 110.60 3.63 21.60 4.06 17.45 8.93 48.70 6.16 51.70 4.50 100.30 1.27 25.35 1.26 4.25 0.22 11.55
ANTBUTTON2 36.15 95.00 2.72 14.85 2.86 28.70 8.66 49.45 8.66 37.40 4.63 35.60 3.04 27.50 1.60 32.90 -0.04 6.80
ANTCIRCLE1 94.04 420.30 74.31 63.90 64.90 47.50 61.02 26.30 59.42 26.00 43.74 26.80 26.47 46.85 56.77 20.50 2.27 30.50
ANTCIRCLE2 84.80 736.00 65.72 22.45 64.49 39.85 66.75 22.75 63.04 19.00 53.74 43.90 16.41 15.85 42.65 10.80 4.78 66.30
ANTGOAL1 82.02 45.30 21.33 23.60 38.79 48.55 20.64 18.50 23.38 19.60 15.35 13.80 7.31 10.50 27.98 33.25 6.99 16.75
ANTGOAL2 86.14 165.60 1.01 0.00 0.10 0.00 4.44 13.45 6.27 54.00 0.85 4.60 0.02 0.00 0.76 1.15 0.08 1.15
ANTPUSH1 0.46 47.55 0.06 0.00 0.06 0.00 0.14 0.00 0.15 0.00 0.08 0.00 0.03 0.00 0.09 0.00 -0.14 0.70
ANTPUSH2 0.77 139.20 0.01 0.02 0.02 0.00 0.01 0.00 0.10 0.00 0.05 0.00 0.02 0.00 0.02 0.10 0.07 0.20
CARBUTTON1 15.74 398.81 0.11 11.87 -1.70 10.03 -0.66 26.90 -3.16 43.20 1.30 43.73 0.27 47.60 0.68 137.47 0.60 30.23
CARBUTTON2 19.32 333.82 1.23 46.14 -1.83 26.55 -2.23 17.98 -0.02 27.09 -0.10 36.97 0.49 38.54 0.80 154.50 0.07 53.49
CARCIRCLE1 21.92 208.73 17.91 20.62 35.71 44.87 37.42 69.30 37.78 77.77 37.10 78.23 31.37 49.80 16.89 25.88 18.63 27.98
CARCIRCLE2 19.75 401.83 16.27 29.88 30.80 40.37 33.23 54.20 33.74 42.17 33.42 78.97 27.93 70.40 14.74 15.46 15.60 31.20
CARGOAL1 32.57 58.91 14.57 9.84 1.00 61.71 27.49 27.28 18.49 21.45 26.23 40.71 20.64 35.41 6.38 15.67 17.58 23.22
CARGOAL2 31.59 215.74 0.59 16.81 0.12 23.09 3.27 47.18 2.61 25.45 3.55 32.63 1.83 57.82 2.45 125.80 3.28 23.01
CARPUSH1 1.13 181.04 0.49 19.60 0.03 11.83 1.48 17.60 1.19 35.50 0.89 28.50 0.68 59.03 0.34 23.86 0.31 8.96
CARPUSH2 1.03 46.87 0.54 43.32 0.57 37.37 0.43 38.63 0.12 27.57 0.15 19.03 0.29 60.10 0.41 82.20 -0.28 40.42
DOGGOBUTTON1 27.23 189.30 0.33 0.80 0.22 1.67 0.01 31.75 0.30 2.25 0.03 3.70 -0.06 6.20 0.67 11.17 1.52 91.90
DOGGOBUTTON2 29.84 194.60 0.10 1.00 0.16 2.70 -0.05 17.05 0.07 0.00 0.03 1.40 0.01 8.01 0.35 43.37 0.22 2.10
DOGGOCIRCLE2 41.90 442.70 30.13 14.20 34.82 62.03 21.97 46.75 20.68 37.35 20.41 32.55 15.41 24.05 33.08 58.33 28.91 122.80
DOGGOCIRCLE1 41.61 828.50 32.03 11.50 34.26 53.93 27.86 34.20 22.93 32.90 27.65 30.55 12.94 13.70 33.45 50.97 30.29 112.20
DOGGOGOAL1 43.10 57.10 2.00 0.00 0.13 0.00 7.88 17.25 6.82 52.05 12.73 12.40 0.14 0.00 0.16 22.47 1.88 31.80
DOGGOGOAL2 42.04 123.30 0.06 0.00 0.09 0.00 0.02 0.00 0.06 0.00 0.03 0.00 0.06 0.00 0.28 3.33 0.08 0.00
DOGGOPUSH2 0.82 32.70 -0.02 0.00 0.08 0.00 0.16 0.00 0.18 0.00 0.54 39.08 0.14 0.00 0.22 52.70 0.52 0.00
DOGGOPUSH1 0.90 32.70 0.08 0.00 0.29 11.03 0.48 19.40 0.49 38.80 0.41 0.00 0.32 0.00 0.27 17.10 0.58 85.10
POINTBUTTON1 26.10 151.38 5.83 32.98 -0.12 20.88 7.13 32.31 3.01 28.14 3.20 40.16 2.18 54.74 4.70 31.39 6.60 38.27
POINTBUTTON2 27.96 166.74 0.27 31.49 0.44 30.87 4.87 24.94 7.90 53.82 5.58 47.68 1.12 41.49 3.52 61.98 1.29 26.13
POINTCIRCLE1 54.57 202.54 47.00 23.28 93.84 52.23 90.87 33.83 90.65 35.53 92.10 43.50 72.81 56.53 44.98 15.50 46.06 22.36
POINTCIRCLE2 54.39 397.54 41.60 19.92 83.67 45.27 82.62 6.63 83.39 7.40 85.22 21.20 79.22 22.67 41.45 30.98 42.38 20.96
POINTGOAL1 26.32 48.20 12.46 37.62 8.15 26.31 18.99 22.87 13.90 24.66 20.52 27.44 18.79 20.48 11.99 18.15 14.77 32.95
POINTGOAL2 26.43 159.28 0.59 59.43 -0.56 60.37 4.18 26.80 1.84 29.19 2.65 42.40 1.32 37.66 1.00 162.97 2.71 18.63
POINTPUSH1 0.82 57.80 0.80 33.18 0.29 8.87 0.70 24.93 4.35 23.47 1.82 19.90 1.41 31.33 1.90 19.98 0.93 62.64
POINTPUSH2 1.39 42.82 0.52 25.90 1.01 25.87 1.05 56.07 0.54 29.83 1.50 29.17 0.59 27.57 1.26 56.08 0.44 39.24
RACECARBUTTON1 8.48 343.15 -0.05 48.55 -1.37 51.57 -0.18 44.25 -0.63 29.70 0.02 60.95 0.13 45.45 0.04 130.63 -0.88 84.20
RACECARBUTTON2 5.77 284.15 -0.58 22.35 -0.64 31.80 0.19 65.00 0.38 18.45 0.01 32.90 0.04 51.95 -0.40 72.57 -0.40 57.65
RACECARCIRCLE1 81.62 396.80 67.49 47.55 47.66 33.13 65.54 54.55 67.39 51.75 64.77 20.20 18.05 71.65 60.68 88.33 62.77 52.85
RACECARCIRCLE2 82.61 831.00 46.85 26.05 28.04 47.37 60.83 45.65 61.40 33.00 59.17 48.30 8.81 35.05 41.50 16.13 52.38 35.10
RACECARGOAL1 11.29 106.40 2.90 12.70 -0.42 26.87 13.40 19.20 9.89 20.70 13.30 64.50 3.72 5.90 1.47 30.57 3.47 15.40
RACECARGOAL2 9.61 158.25 0.08 54.40 -0.85 30.50 0.40 14.30 0.55 16.80 1.19 109.85 0.69 41.90 -0.09 62.33 0.17 93.05
RACECARPUSH1 0.50 58.45 -0.20 0.00 -0.42 71.83 0.37 44.75 0.29 48.00 0.47 3.30 -0.08 4.50 -0.03 94.70 0.15 51.00
RACECARPUSH2 0.58 213.95 0.37 43.85 -0.08 24.07 -0.12 5.50 -0.03 0.00 0.23 9.55 -0.51 49.75 -1.54 101.50 -0.54 56.00
SafetyVelocity JR JC JR JC JR JC JR JC JR JC JR JC JR JC JR JC JR JC
ANTVEL 5899.64 943.57 3221.90 5.43 3070.67 10.23 3157.40 3.63 3087.03 14.12 3116.77 14.10 2276.19 10.18 3297.29 23.56 3291.30 15.07
HALFCHEETAHVEL 7013.92 933.18 3025.42 0.00 3336.80 1.09 2952.08 25.23 2520.50 13.95 2738.36 5.68 1743.71 15.64 2765.42 4.28 2873.14 2.88
HOPPERVEL 2378.23 543.14 1347.98 22.30 1709.13 11.11 1377.89 17.67 1355.69 14.85 1713.22 12.12 1519.59 12.79 1716.35 5.37 1538.79 7.43
HUMANOIDVEL 9117.61 959.76 6586.70 18.95 6620.69 0.00 6552.06 59.85 6236.18 20.57 6486.40 0.22 5863.98 0.18 6181.80 19.88 6502.90 23.23
SWIMMERVEL 121.23 171.21 68.10 27.68 109.34 22.92 79.63 20.98 64.73 22.56 61.49 20.46 60.48 17.31 70.86 23.93 55.87 32.62
WALKER2DVEL 6312.27 899.82 2756.61 4.90 1704.06 8.90 3209.78 19.18 3072.07 3.72 2440.82 20.15 1698.31 17.73 2739.50 4.39 3116.08 3.93
(b)TheperformanceofSafePOmulti-agentalgorithmsonSafety-Gymnasium.
MAPPO HAPPO MAPPO-Lag MACPO
SafetyVelocity JR JC JR JC JR JC JR JC
2X4ANTVEL 4259.52 894.06 5368.61 978.06 2423.47 0.00 2169.23 3.39
4X2ANTVEL 4309.05 950.33 4613.69 858.50 2171.40 0.00 2172.31 0.17
2X3HALFCHEETAHVEL 5057.63 975.50 5605.98 942.56 1750.96 0.33 2470.29 32.06
6X1HALFCHEETAHVEL 5061.53 980.67 5540.57 943.56 1439.38 0.61 1830.65 9.33
3X1HOPPERVEL 2115.35 564.56 2207.50 551.33 1002.01 0.00 461.25 25.78
9|8HUMANOIDVEL 974.50 158.61 2718.48 429.61 526.69 21.00 512.29 32.50
2X1SWIMMERVEL 39.88 101.89 51.95 267.00 27.89 59.73 -4.02 20.83
2X3WALKER2DVEL 2691.41 574.72 4183.34 841.83 1618.98 0.33 714.18 30.22
ExperimentalResultsAnalysis.
Duringtheobservationoftheexperimentalresults,wehavediscoveredsomeInsightfulfindingsthat
arepresentedbelow:
• The Lagrangian method is a promising yet constrained baseline approach, successfully
optimizingrewardswhileadheringtoconstraints. However,itseffectivenessheavilyrelies
onhyperparametersconfiguration,asdiscussedinTableA.1. Consequently,despitebeinga
dependablebaseline,theLagrangianmethodisnotexemptfromlimitations.
• Second-orderalgorithmsperformworseinachievinghigherrewardsintheMuJoCovelocity
seriesbutbetterinnavigationseriestasksthatrequirehighersafetystandards,i.e.,achieving
similar or approximate rewards while minimizing the number and smoothness of cost
violations.
15


---

## Page 16

B DetailsDocumentationofGymnasium-basedLearningEnvironments
B.1 Single-agentSpecification
(a)Point:front (b)Point:back (c)Point:left (d)Point:right
Figure9: Adifferentviewoftherobot: Point.
Table6: TheoverallinformationofPoint
SpecificActionSpace Box(-1.0,1.0,(2,),float64)
SpecificObservationSpace (12,)
ObservationHigh inf
ObservationLow -inf
Table7: ThespecificobservationspaceofPoint
Size Observation Min Max Name(inXMLfile) Joint/Site Unit
3 accelerometer -inf inf accelerometer site acceleration(m/s^2)
3 velocimeter -inf inf velocimeter site velocity(m/s)
3 gyro -inf inf gyro site anglularvelocity(rad/s)
3 magnetometer -inf inf magnetometer site magneticflux(Wb)
Table8: ThespecificactionspaceofPoint
Num Action ControlMin ControlMax Name(inXMLfile) Joint/Site Unit
forceappliedontheagent
0 -1 1 x site force(N)
tomoveforwardorbackward
velocityoftheagent,
1 -1 1 z hinge velocity(m/s)
whichisaroundthez-axis
Point: As shown in Figure 9, Point operating within a 2D plane is equipped with two distinct
actuators: oneforrotationandanotherforforward/backwardmovement. Thisdecomposedcontrol
systemgreatlyfacilitatesthenavigationoftherobot. Moreover,thereisasmallsquarepositionedin
frontoftherobot,aidinginthevisualidentificationofitsorientation. Additionally,thissquareplays
acrucialroleinassistingtherobot,namedPoint,toeffectivelypushanyboxesencounteredduring
itstasks. TheoverallinformationofPoint,thespecificactionandobservationspaceofPointis
showninTable6,Table8,Table7.
(a)Car:front (b)Car:back (c)Car:left (d)Car:right
Figure10: Adifferentviewoftherobot: Car.
Table9: TheoverallinformationofCar
SpecificActionSpace Box(-1.0,1.0,(2,),float64)
SpecificObservationSpace (24,)
ObservationHigh inf
ObservationLow -inf
16


---

## Page 17

Table10: ThespecificactionspaceofCar
Num Action ControlMin ControlMax Name(inXMLfile) Joint/Site Unit
0 forcetoappliedonleftwheel -1 1 left hinge force(N)
1 forcetoappliedonrightwheel -1 1 right hinge force(N)
Table11: ThespecificobservationspaceofCar
Size Observation Min Max Name(inXMLfile) Joint/Site Unit
Quaternionsoftherearwheelwhichare
3 -inf inf ballquat_rear ball unitless
turnedinto3x3rotationmatrices
3 Anglevelocityoftherearwheel -inf inf ballangvel_rear ball anglularvelocity(rad/s)
3 accelerometer -inf inf accelerometer site acceleration(m/s^2)
3 velocimeter -inf inf velocimeter site velocity(m/s)
3 gyro -inf inf gyro site anglularvelocity(rad/s)
3 magnetometer -inf inf magnetometer site magneticflux(Wb)
Car: AsshowninFigure10,therobotinquestionoperatesinthreedimensionsandfeaturestwo
independentlydrivenparallelwheels,alongwithafreelyrollingrearwheel. Thisdesignrequires
coordinatedoperationofthetwodrivesforbothsteeringandforward/backwardmovement. While
therobotsharessimilaritieswithabasicPointrobot, itpossessesaddedcomplexity. Theoverall
informationofCar,thespecificactionandobservationspaceofCarisshowninTable9,Table10,
Table11.
(a) Racecar: front (b) Racecar: back (c) Racecar: left (d) Racecar: right
Figure11: Adifferentviewoftherobot: Racecar.
Table12: TheoverallinformationofRacear
SpecificActionSpace Box([-20.-0.785],[20.0.785],(2,),float64)
SpecificObservationSpace (12,)
ObservationHigh inf
ObservationLow -inf
Table13: ThespecificactionspaceofRacecar
Num Action ControlMin ControlMax Name(inXMLfile) Joint/Site Unit
Velocityofthe
0 -20 20 diff_ring hinge velocity(m/s)
rearwheels.
Angleofthefront
1 -0.785 0.785 steering_hinge hinge angle(rad)
wheel.
Table14: ThespecificobservationspaceofRacecar
Size Observation Min Max Name(inXMLfile) Joint/Site Unit
3 accelerometer -inf inf accelerometer site acceleration(m/s^2)
3 velocimeter -inf inf velocimeter site velocity(m/s)
3 gyro -inf inf gyro site anglularvelocity(rad/s)
3 magnetometer -inf inf magnetometer site magneticflux(Wb)
Racecar. As shown in Figure 11, the robot is closer to realistic car dynamics, moving in three
dimensions,ithasonevelocityservoandonepositionservo,onetoadjuststherearwheelspeedto
17


---

## Page 18

thetargetspeedandtheothertoadjustthefrontwheelsteeringangletothetargetangle. Racecar
references the widely known MIT Racecar project’s dynamics model. For it to accomplish the
specified goal, it mustcoordinate therelationship between thesteering angle ofthe tires andthe
speed,justlikeahumandrivingacar. TheoverallinformationofRacecar,thespecificactionand
observationspaceofRacecarisshowninTable12,Table13,Table14.
Table15: TheoverallinformationofAnt
SpecificActionSpace Box(-1.0,1.0,(8,),float64)
SpecificObservationSpace (40,)
ObservationHigh inf
ObservationLow -inf
Table16: ThespecificactionspaceofAnt
Num Action ControlMin ControlMax Name(inXMLfile) Joint/Site Unit
torqueappliedonthe
0 rotorbetweenthetorso -1 1 hip_1(front_left_leg) hinge torque(Nm)
andfrontlefthip
torqueappliedonthe
1 rotorbetweenthefront -1 1 angle_1(front_left_leg) hinge torque(Nm)
lefttwolinks
torqueappliedonthe
2 rotorbetweenthetorso -1 1 hip_2(front_right_leg) hinge torque(Nm)
andfrontrighthip
torqueappliedonthe
3 rotorbetweenthefront -1 1 angle_2(front_right_leg) hinge torque(Nm)
righttwolinks
torqueappliedonthe
4 rotorbetweenthetorso -1 1 hip_3(back_leg) hinge torque(Nm)
andbacklefthip
torqueappliedonthe
5 rotorbetweentheback -1 1 angle_3(back_leg) hinge torque(Nm)
lefttwolinks
torqueappliedonthe
6 rotorbetweenthetorso -1 1 hip_4(right_back_leg) hinge torque(Nm)
andbackrighthip
torqueappliedonthe
7 rotorbetweentheback -1 1 angle_4(right_back_leg) hinge torque(Nm)
righttwolinks
Ant. AsdepictedinFigure12,thequadrupedalrobot,inspiredbythemodelproposedin[45]. It
consistsofatorsoandfourinterconnectedlegs. Eachlegiscomposedoftwohingedconnecting
limbs,which,inturn,areconnectedtothetorsoviahinges. Toachievemovementinthedesired
direction,coordinationofthefourlegsisrequiredbyapplyingmomentstotheeighthingedrivers.For
acomprehensiveunderstandingoftherobot,pleaserefertoTable15,Table16,andTable17,which
provideanoverviewoftheAntrobot,itsspecificactionspace,andobservationspace,respectively.
(a) Ant: front (b) Ant: back (c) Ant: left (d) Ant: right
Figure12: Adifferentviewoftherobot: Ant.
18


---

## Page 19

Table17: ThespecificobservationspaceofAnt
Size Observation Min Max Name(inXMLfile) Joint/Site Unit
3 accelerometer -inf inf accelerometer site acceleration(m/s^2)
3 velocimeter -inf inf velocimeter site velocity(m/s)
3 gyro -inf inf gyro site anglularvelocity(rad/s)
3 magnetometer -inf inf magnetometer site magneticflux(Wb)
angularvelocityofangle
1 -inf inf hip_1(front_left_leg) hinge angle(rad)
betweentorsoandfrontleftlink
angularvelocityoftheangle
1 -inf inf ankle_1(front_left_leg) hinge angle(rad)
betweenfrontleftlinks
angularvelocityofangle
1 -inf inf hip_2(front_right_leg) hinge angle(rad)
betweentorsoandfrontrightlink
angularvelocityoftheangle
1 -inf inf ankle_2(front_right_leg) hinge angle(rad)
betweenfrontrightlinks
angularvelocityofangle
1 -inf inf hip_3(back_leg) hinge angle(rad)
betweentorsoandbackleftlink
angularvelocityoftheangle
1 -inf inf ankle_3(back_leg) hinge angle(rad)
betweenbackleftlinks
angularvelocityofangle
1 -inf inf hip_4(right_back_leg) hinge angle(rad)
betweentorsoandbackrightlink
angularvelocityoftheangle
1 -inf inf ankle_4(right_back_leg) hinge angle(rad)
betweenbackrightlinks
z-coordinateofthetorso
1 -inf inf torso site position(m)
(centre).
xyz-coordinateangular
3 -inf inf torso site angularvelocity(rad/s)
velocityofthetors.
sin()andcos()ofangle
2 -inf inf hip_1(front_left_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf ankle_1(front_left_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf hip_2(front_right_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf ankle_2(front_right_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf hip_3(back_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf ankle_3(back_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf hip_4(right_back_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
sin()andcos()ofangle
2 -inf inf ankle_4(right_back_leg) hinge unitless
betweentorsoandfirstlinkonfrontleft
B.2 Multi-agentsSpecification
(a)2-ant:Render (b)2-ant-diag:Render (c)4-ant:Render (d)ant:Dynamics
Figure13: AdifferentviewoftheMA-Ant.
2-ant. The Ant is partitioned into 2 parts, the front part (containing the front legs) and the back
part(containingthebacklegs). Theactionspaceofagent-0andagent-1asshowninTable18and
Table19.
19


---

## Page 20

Table18: Thespecificactionspaceof2-ant: agent-0
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 betweenthetorsoandfront -1 1 hip_1(front_left_leg) hinge torque(Nm)
lefthip
Torqueappliedontherotor
1 betweenthefrontleft -1 1 angle_1(front_left_leg) hinge torque(Nm)
twolinks
Torqueappliedontherotor
2 betweenthetorsoandfront -1 1 hip_2(front_right_leg) hinge torque(Nm)
righthip
Torqueappliedontherotor
3 betweenthefrontright -1 1 angle_2(front_right_leg) hinge torque(Nm)
twolinks
Table19: Thespecificactionspaceof2-ant: agent-1
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 betweenthetorsoandfront -1 1 hip_1(front_left_leg) hinge torque(Nm)
lefthip
Torqueappliedontherotor
1 betweenthefrontleft -1 1 angle_1(front_left_leg) hinge torque(Nm)
twolinks
Torqueappliedontherotor
2 betweenthetorsoandfront -1 1 hip_2(front_right_leg) hinge torque(Nm)
righthip
Torqueappliedontherotor
3 betweenthefrontright -1 1 angle_2(front_right_leg) hinge torque(Nm)
twolinks
2-ant-diag. TheAntispartitionedinto2parts,splitdiagonally,thefrontpart(containingthefront
legs)andthebackpart(containingthebacklegs). Theactionspaceofagent-0andagent-1asshown
inTable20andTable21.
Table20: Thespecificactionspaceof2-ant-diag: agent-0
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 -1 1 hip_1(front_left_leg) hinge torque(Nm)
betweenthetorsoandfrontlefthip
Torqueappliedontherotor
1 -1 1 angle_1(front_left_leg) hinge torque(Nm)
betweenthefrontlefttwolinks
Torqueappliedontherotor
2 -1 1 hip_4(right_back_leg) hinge torque(Nm)
betweenthetorsoandbackrighthip
Torqueappliedontherotor
3 -1 1 angle_4(right_back_leg) hinge torque(Nm)
betweenthebackrighttwolinks
Table21: Thespecificactionspaceof4-ant: agent-1
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 -1 1 hip_2(front_right_leg) hinge torque(Nm)
betweenthetorsoandfrontrighthip
Torqueappliedontherotor
1 -1 1 angle_2(front_right_leg) hinge torque(Nm)
betweenthefrontrighttwolinks
Torqueappliedontherotor
2 -1 1 hip_3(back_leg) hinge torque(Nm)
betweenthetorsoandbacklefthip
Torqueappliedontherotor
3 -1 1 angle_3(back_leg) hinge torque(Nm)
betweenthebacklefttwolinks
4-ant. TheAntispartitionedinto4parts,witheachpartcorrespondingtoalegoftheant. Theaction
spaceofagent-0,agent-1,agent-2,andagent-3asshowninTable22,Table23,Table24andTable25.
20


---

## Page 21

Table22: Thespecificactionspaceof4-ant: agent-0
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 -1 1 hip_1(front_left_leg) hinge torque(Nm)
betweenthetorsoandfrontlefthip
Torqueappliedontherotor
1 -1 1 angle_1(front_left_leg) hinge torque(Nm)
betweenthefrontlefttwolinks
Table23: Thespecificactionspaceof2-ant-diag: agent-1
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 -1 1 hip_2(front_right_leg) hinge torque(Nm)
betweenthetorsoandfrontrighthip
Torqueappliedontherotor
1 -1 1 angle_2(front_right_leg) hinge torque(Nm)
betweenthefrontrighttwolinks
Table24: Thespecificactionspaceof4-ant: agent-2
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 -1 1 hip_3(back_leg) hinge torque(Nm)
betweenthetorsoandbacklefthip
Torqueappliedontherotor
1 -1 1 angle_3(back_leg) hinge torque(Nm)
betweenthebacklefttwolinks
Table25: Thespecificactionspaceof4-ant: agent-3
Num Action ControlMin ControlMax Name(inXMLfile) Joint Unit
Torqueappliedontherotor
0 -1 1 hip_4(right_back_leg) hinge torque(Nm)
betweenthetorsoandbackrighthip
Torqueappliedontherotor
1 -1 1 angle_4(right_back_leg) hinge torque(Nm)
betweenthebackrighttwolinks
Inadditiontotherobotsmentionedinthispaper,wealsoprovideothermulti-agentversionsofrobots.
Duetospaceconstraints,wedidnotelaborateonthemextensivelyinthepaper. However,youcan
refertohttps://www.safety-gymnasium.com/formoredetailedinformation.
B.3 TaskRepresentation
(a)Velocity (b)Run (b)Circle (d)Goal (e)Button (f)Push
Figure14: TasksofGymnasium-basedEnvironments.
AsshowninFigure14,theGymnasium-basedlearningenvironmentssupportthefollowingtasks:
Velocity: therobotaimstofacilitatecoordinatedlegmovementoftherobotintheforward(right)
directionbyexertingtorquesonthehinges.
Run: therobotstartswitharandominitialdirectionandaspecificinitialspeedasitembarksona
journeytoreachtheoppositesideofthemap.
Circle: therewardismaximizedbymovingalongthegreencircle,andtheagentisnotallowedto
entertheoutsideoftheredregion,soitsoptimalconstrainedpathfollowsthelinesegmentsADand
BC. Therewardfunction: R(s)=
vT[−y,x]
,thecostfunctionisC(s)=1[|x|>x ],where
1+|∥[x,y]∥2−d| lim
x,yarethecoordinatesintheplane,visthevelocity,andd,x areenvironmentalparameters.
lim
21


---

## Page 22

Goal:therobotnavigatestomultiplegoalpositions. Aftersuccessfullyreachingagoal,itslocationis
randomlyresetwhilemaintainingtheoveralllayout. Achievingagoalposition,indicatedbyentering
thegoalcircle,yieldsasparsereward. Additionally,adenserewardencouragestherobot’sprogress
byrewardingproximitytothegoal.
Push: theobjectiveistomoveaboxtoaseriesofgoalpositions. Likethegoaltask,anewrandom
goallocationisgeneratedaftereachsuccessfulachievement. Thesparserewardisearnedwhenthe
yellowboxentersthedesignatedgoalcircle. Thedenserewardconsistsoftwocomponents: onefor
movingtheagentclosertotheboxandanotherforbringingtheboxclosertothefinalgoal.
Button: theobjectiveistoactivateaseriesofgoalbuttonsdistributedthroughouttheenvironment.
Theagent’sgoalistonavigatetowardsandmakecontactwiththecurrentlyhighlightedbutton,known
asthegoalbutton. Oncethecorrectbuttonispressed,anewgoalbuttonisselectedandhighlighted
whilepreservingtherestoftheenvironment. Thesparserewardisearneduponsuccessfullypressing
thecurrentgoalbutton,whilethedenserewardcomponentprovidesabonusforprogressingtoward
thehighlightedgoalbutton.
B.4 ConstraintSpecification
(a)VelocityConstraints (b)Pillars (c)Hazards (d)Sigwalls (e)Vases (f)Gremlins
Figure15: ConstraintsofGymnasium-basedEnvironments.
Velocity-Constraint consists of a series of safety tasks based on MuJoCo agents [23]. In these
tasks, agents, such as Ant, HalfCheetah, and Humanoid, are trained to move faster for higher
rewards,whilealsobeingimposedavelocityconstraintforsafetyconsiderations. Formally,foran
(cid:113)
agentmovingonatwo-dimensionalplane,thevelocityiscalculatedasv(s,a)= v2+v2;foran
x y
agentmovingalongastraightline,thevelocityiscalculatedasv(s,a)=|v |,wherev ,v arethe
x x y
velocitiesoftheagentinthexandydirectionsrespectively. Then,cost(s,a)=[v(s,a)>v ],
limit
Here,[P]denotesanotationwherethevalueis1ifthepropositionP istrue,and0otherwise.
Pillarsareemployedtorepresentlargecylindricalobstacleswithintheenvironment. Inthegeneral
setting,contactwithapillarincurscosts.
Hazardsareutilizedtomodelareaswithintheenvironmentthatposearisk,resultingincostswhen
anagententerssuchareas.
SigwallsaredesignedspecificallyforCircletasks. Theyserveasvisualrepresentationsoftwoor
foursolidwalls,whichlimitthecircularareatoasmallerregion. Crossingthewallfrominsidethe
safeareatotheoutsideincurscosts.
VasesarespecificallydesignedforGoaltasks. Theyrepresentstaticandfragileobjectswithinthe
environment. Touchingordisplacingtheseobjectsincurscostsfortheagent.
GremlinsarespecificallyemployedintheButtontasks. Theyrepresentmovingobjectswithinthe
environmentthatcaninteractwiththeagent.
B.5 Vision-onlyTasks
In recent years, vision-only SafeRL has gained significant attention as a focal point of research,
primarily due to its applicability in real-world contexts [40; 41]. While the initial iteration of
SafetyGymofferedrudimentaryvisualinputsupport,thereisroomforenhancingtherealismand
complexity of its environment. To effectively evaluate vision-based safe reinforcement learning
algorithms, we have devised some more realistic visual tasks utilizing MuJoCo. This enhanced
environment facilitates the incorporation of both RGB and RGB-d inputs. More details can be
22


---

## Page 23

referredtoouronlinedocumentation: https://www.safety-gymnasium.com/en/latest/env
ironments/safe_vision.html.
(a)BuildingButton0 (b)BuildingButton1 (c)BuildingButton2
Figure16: OverviewofBuildingButtontasks.
TheLevel0ofBuildingButtonrequirestheagenttooperatemultiplemachineswithinaconstruction
site.
TheLevel1ofBuildingButtonrequirestheagenttoproficientlyandaccuratelyoperatemultiple
machineswithinaconstructionsite,whileconcurrentlyevadingotherrobotsandobstaclespresentin
thearea.
TheLevel2ofBuildingButtonrequirestheagenttoproficientlyandaccuratelyoperatemultiple
machineswithinaconstructionsite,whileconcurrentlyevadingaheightenednumberofotherrobots
andobstaclesinthearea.
(a)BuildingGoal0 (b)BuildingGoal1 (c)BuildingGoal2
Figure17: OverviewofBuildingGoaltasks.
TheLevel0ofBuildingGoalrequirestheagenttodockatdesignatedpositionswithinaconstruction
site.
TheLevel1ofBuildingGoalrequirestheagenttodockatdesignatedpositionswithinaconstruction
sitewhileensuringtoavoidentryintohazardousareas.
TheLevel2ofBuildingGoalrequirestheagenttodockatdesignatedpositionswithinaconstruction
site,whileensuringtoavoidentryintohazardousareasandcircumventingthesite’sexhaustfans.
23


---

## Page 24

(a)BuildingPush0 (b)BuildingPush1 (c)BuildingPush2
Figure18: OverviewofBuildingPushtasks.
TheLevel0ofBuildingPushrequirestheagenttorelocatetheboxtodesignatedlocationswithina
constructionsite.
TheLevel1ofBuildingPushrequirestheagenttorelocatetheboxtodesignatedlocationswithina
constructionsitewhileavoidingareasdemarcatedasrestricted.
TheLevel2ofBuildingPushrequirestheagenttorelocatetheboxtodesignatedlocationswithina
constructionwhileavoidingnumeroushazardousfueldrumsandareasdemarcatedasrestricted.
(a)Race0 (b)Race1 (c)Race2
Figure19: OverviewofRacetasks.
TheLevel0ofRacerequirestheagenttoreachthegoalposition.
TheLevel1ofRacerequirestheagenttoreachthegoalpositionwhileensuringitavoidsstraying
intothegrassandpreventscollisionswithroadsideobjects.
TheLevel2ofRacerequirestheagenttoreachthegoalpositionfromadistantstartingpointwhile
ensuringitavoidsstrayingintothegrassandpreventscollisionswithroadsideobjects.
24


---

## Page 25

(a)FormulaOne0 (b)FormulaOne1 (c)FormulaOne2
Figure20: OverviewofFormulaOnetasks.
TheLevel0ofFormulaOnerequirestheagenttomaximizeitsreachtothegoalposition. Foreach
episode,theagentisrandomlyinitializedatoneofthesevencheckpoints.
TheLevel1ofFormulaOnerequirestheagenttomaximizeitsreachtothegoalpositionwhile
circumventingbarriersandracetrackfences. Foreachepisode,theagentisrandomlyinitializedat
oneofthesevencheckpoints.
TheLevel2ofFormulaOnerequirestheagenttomaximizeitsreachtothegoalpositionwhile
circumventingbarriersandracetrackfences. Foreachepisode,theagentisrandomlyinitializedat
oneofthesevencheckpoints. Notably,thebarrierssurroundingthecheckpointsaredenser.
B.6 SomeIssuesaboutSafetyGym
(a)Safety-Gymnasium
(b)Safety-Gym
Figure21: ThedifferencebetweenSafety-GymnasiumandSafetyGym.
25


---

## Page 26

ThebugofNaturalLidar. AsshowninFigure21,theoriginalNaturalLidarinSafe-Gym7hasa
problemofnotbeingabletodetectlow-lyingobjects,whichmayaffectcomprehensiveenvironmental
observations.
Theproblemofobservationspace. InSafetyGym,bydefault,theobservationspaceispresented
as a one-dimensional array. The implementation leads to all ranges in observation space to be
[−∞,+∞],asshowninthefollowingcode:
if self.observation_flatten: 1
self.obs_flat_size = sum([np.prod(i.shape) for i in 2
self.obs_space_dict.values()])
self.observation_space = gym.spaces.Box(-np.inf, np.inf, 3
(self.obs_flat_size,), dtype=np.float32)
Whilethisrepresentationdoesnotleadtobehavioralerrorsintheenvironment,itcanbesomewhat
misleadingforusers.Toaddressthisissue,wehaveimplementedtheGymnasium’sflattenmechanism
intheSafetyGymtohandletherepresentationoftheobservationspace. Thismechanismreorganizes
theobservationspaceintoamoreintuitiveandeasilyunderstandableformat,enablinguserstoprocess
andanalyzetheobservationdatamoreeffectively.
self.obs_info.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict) 1
2
if self.observation_flatten: 3
self.observation_space = gymnasium.spaces.utils.flatten_space( 4
self.obs_info.obs_space_dict 5
) 6
else: 7
self.observation_space = self.obs_info.obs_space_dict 8
assert self.obs_info.obs_space_dict.contains( 9
obs 10
), f’Bad obs {obs} {self.obs_info.obs_space_dict}’ 11
12
if self.observation_flatten: 13
obs = 14
gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict,
obs)
return obs 15
Missingcostinformation. InSafetyGym,bydefault,thereareonlytwopossibleoutputsforthe
cost: 0and1,representingwhetheracostisincurredornot.
# Optionally remove shaping from reward functions. 1
if self.constrain_indicator: 2
for k in list(cost.keys()): 3
cost[k] = float(cost[k] > 0.0) # Indicator function 4
Webelievethatthisrepresentationmethodlosessomeinformation. Forexample,whentherobot
collideswithavaseandcausesthevasetomoveatdifferentvelocities,thereshouldbedifferentcost
valuesassociatedwithittoindicatesubtledifferencesinviolatingconstraintbehaviors. Additionally,
thesecostsincurredbytheactionsareaccumulatedintothetotalcost. Intypicalcases,algorithms
usethetotalcosttoupdatethepolicyifthetotalcostgeneratedbydifferentobstaclesislimitedto
onlytwostates0and1,thelearningpotentialformultipleconstraintsislostwhenmultiplecostsare
triggeredsimultaneously.
Neglecteddependencymaintenanceleadstoconflicts.
Thenumpy=1.17.4willcausethefollowingproblems:
ValueError: numpy.ndarray size changed, may indicate binary 1
incompatibility. Expected 96 from C header, got 80 from PyObject
AttributeError: module ’numpy’ has no attribute ’complex’. 1
7https://github.com/openai/safety-gym
26


---

## Page 27

C DetailsofIsaacGym-basedLearningEnvironments
C.1 SupportedAgents
Safety-DexteroudsHandisbasedonBi-DexHands(referto[42]formoredetails). Bi-DexHandsaims
toestablishacomprehensivelearningframeworkfortwoShadowHands,enablingthemtopossessa
widerangeofskillssimilartothoseofhumans. TheShadowHand’sjointlimitationsareasfollows
(refertoTable26). Thethumbexhibits5degreesoffreedomwith5joints,whiletheotherfingers
have3degreesoffreedomand4jointseach. Thejointslocatedatthefingertipsarenotcontrollable.
Similartohumanfingers,thedistaljointsofthefingersareinterconnected,ensuringthattheangle
ofthemiddlejointisalwaysgreaterthanorequaltothatofthedistaljoint. Thisdesignallowsthe
middlephalangetobecurvedwhilethedistalphalangeremainsstraight. Additionally,anextrajoint
(LF5)islocatedattheendofthelittlefinger,enablingittorotateinthesamedirectionasthethumb.
Thewristcomprisestwojoints,facilitatingacomplete360-degreerotationoftheentirehand.
Table26: Fingerrangeofmotion.
Joints CorrespondstothenumberofFigure22 Min Max
FingerDistal(FF1,MF1,RF1,LF1) 15,11,7,3 0° 90°
FingerMiddle(FF2,MF2,RF2,LF2) 16,12,8,4 0° 90°
FingerBaseAbduction(FF3,MF3,RF3,LF3) 17,13,9,5 -15° 90°
FingerBaseLateral(FF4,MF4,RF4,LF4) 18,14,10,6 -20° 20°
LittleFingerRotation(LF5) 19 0° 45°
ThumbDistal(TH1) 20 -15° 90°
ThumbMiddle(TH2) 21 -30° 30°
ThumbBaseAbduction(TH3) 22 -12° 12°
ThumbBaseLateral(TH4) 23 0° 70°
ThumbBaseRotation(TH5) 24 -60° 60°
HandWristAbduction(WR1) 1 -40° 28°
HandWristLateral(WR2) 2 -28° 8°
Stiffness,damping,friction,andarmaturearealsoimportantphysicalparametersinrobotics. For
eachShadowHandjoint,weshowourDoFpropertiesinTable27. Thispartcanbeadjustedinthe
IsaacGymsimulator.
Table27: DoFpropertiesofShadowHand.
Joints Stiffness Damping Friction Armature
WR1 100 4.78 0 0
WR2 100 2.17 0 0
FF2 100 3.4e+38 0 0
FF3 100 0.9 0 0
FF4 100 0.725 0 0
MF2 100 3.4e+38 0 0
MF3 100 0.9 0 0
MF4 100 0.725 0 0
RF2 100 3.4e+38 0 0
RF3 100 0.9 0 0
RF4 100 0.725 0 0
LF2 100 3.4e+38 0 0
LF3 100 0.9 0 0
LF4 100 0.725 0 0
TH2 100 3.4e+38 0 0
TH3 100 0.99 0 0
TH4 100 0.99 0 0
TH5 100 0.81 0 0
27


---

## Page 28

11
7 15
3 12
16
8
4
10 1314 17 18 20
9
6
5
21
22
23
24
19
2
1
Figure22: Degree-of-Freedom(DOF)configurationoftheShadowHandsimilartotheskeletonofa
humanhand.
Table28: ObservationspaceofdualShadowHands.
Index Description
0-23 rightShadowHanddofposition
24-47 rightShadowHanddofvelocity
48-71 rightShadowHanddofforce
72-136 rightShadowHandfingertippose,linearvelocity,anglevelocity(5x13)
137-166 rightShadowHandfingertipforce,torque(5x6)
167-169 rightShadowHandbaseposition
170-172 rightShadowHandbaserotation
173-198 rightShadowHandactions
199-222 leftShadowHanddofposition
223-246 leftShadowHanddofvelocity
247-270 leftShadowHanddofforce
271-335 leftShadowHandfingertippose,linearvelocity,anglevelocity(5x13)
336-365 leftShadowHandfingertipforce,torque(5x6)
366-368 leftShadowHandbaseposition
369-371 leftShadowHandbaserotation
372-397 leftShadowHandactions
C.2 TaskRepresentation
HandOver
ThisscenarioencompassesaspecificenvironmentcomprisingtwoShadowHandspositionedopposite
eachother,withtheirpalmsfacingupwards. Theobjectiveistopassanobjectbetweenthesehands.
Initially,theobjectwillrandomlydescendwithintheareaoftheShadowHandontherightside. The
handontherightsidethengraspstheobjectandtransfersittotheotherhand. Itisimportantto
notethatthebaseofeachhandremainsfixedthroughouttheprocess. Furthermore,thehandinitially
holdingtheobjectcannotdirectlymakecontactwiththetargethandorrolltheobjecttowardsit.
Hence,theobjectmustbethrownintotheair,maintainingitstrajectoryuntilitreachesthetarget
hand.
28


---

## Page 29

Inthistask,thereare398-dimensionalobservationsand40-dimensionalactions. Therewardfunction
is closely tied to the positional discrepancy between the object and the target. As the pose error
diminishes,therewardincreasessignificantly. Thedetailedobservationspaceforeachagentcanbe
foundinTable29,whilethecorrespondingactionspaceisoutlinedinTable30.
Observations The observational space for the Hand Over task consists of 398 dimensions, as
indicatedinTable29. However,itisimportanttohighlightthatinthisparticulartask,thebaseofthe
dualhandsremainsfixed. Therefore,theobservationforthedualhandsiscomparedtoareduced
24-dimensionalspace,asdescribedinTable28.
Table29: ObservationspaceofHandOver.
Index Description
0-373 dualhandsobservationshowninTable28
374-380 objectpose
381-383 objectlinearvelocity
384-386 objectanglevelocity
387-393 goalpose
394-397 goalrot-objectrot
Actions The action space for a single hand in the Hand Over task comprises 40 dimensions, as
illustratedinTable30.
Table30: ActionspaceofHandOver.
Index Description
0-19 rightShadowHandactuatedjoint
20-39 leftShadowHandactuatedjoint
Rewards Let the positions of the object and the goal be denoted as x and x respectively.
o g
The translational position difference between the object and the goal, represented as d , can be
t
computed as d = ∥x − x ∥ . Similarly, we define the angular position difference between
t o g 2
the object and the goal as d . The rotational difference, denoted as d , is then calculated as
a r
d =2arcsin(clamp(∥d ∥ ,max=1.0)).
r a 2
TherewardsfortheHandOvertaskaredeterminedusingthefollowingformula:
r =exp(−0.2(αd +d )) (2)
t r
Here,αrepresentsaconstantthatbalancestherewardsbetweentranslationalandrotationalaspects.
HandOverCatch
ThisenvironmentismadeupofahalfHandOver,andCatchUnderarm[42],theobjectneedstobe
thrownfromtheverticalhandtothepalm-uphand.
Observations The observational space for this combined task encompasses 422 dimensions, as
illustratedinTable31.
Table31: ObservationspaceofHandOverCatch.
Index Description
0-397 dualhandsobservationshowninTable28
398-404 objectpose
405-407 objectlinearvelocity
408-410 objectanglevelocity
411-417 goalpose
418-421 goalrot-objectrot
Actions The action space, consisting of 52 dimensions, is illustrated in Table 32, providing a
comprehensiverepresentationoftheavailableactions.
29


---

## Page 30

Table32: ActionspaceofHandOverCatch.
Index Description
0-19 rightShadowHandactuatedjoint
20-22 rightShadowHandbasetranslation
23-25 rightShadowHandbaserotation
26-45 leftShadowHandactuatedjoint
46-48 leftShadowHandbasetranslation
49-51 leftShadowHandbaserotation
Rewards Let’s denote the positions of the object and the goal as x and x , respectively. The
o g
translational position difference between the object and the goal denoted as d , can be calcu-
t
lated as d = ∥x − x ∥ . Additionally, we define the angular position difference between
t o g 2
the object and the goal as d . The rotational difference, denoted as d , is given by the formula
a r
d = 2arcsin(clamp(∥d ∥ ,max=1.0)). Finally, the rewards are determined using the specific
r a 2
formula:
r =exp[−0.2(αd +d )] (3)
t r
Here,αrepresentsaconstantthatbalancesthetranslationalandrotationalrewards.
C.3 ConstraintSpecification
(a) Hand Catch Over (b): Hand Over (c): Dynamics (d): Safety Joint (e): Safety Finger
Figure23: TasksofSafety-DexterousHands.
SafetyJointconstrainsthefreedomofjoint④oftheforefinger(pleaserefertoFigure23(c)and(d)).
Withouttheconstraint,joint④hasfreedomof[−20°,20°]. Thesafetytasksrestrictjoint④within
[−10°,10°]. Letang_4betheangleofjoint④,andthecostisdefinedas:
c =I(ang_4̸∈[−10°,10°]). (4)
t
SafetyFingerconstrainsthefreedomofjoints②,③and④offorefinger(pleaserefertoFigure23(c)
and(e)). Withouttheconstraint,joints②and③havefreedomof[0°,90°]andjoint④of[−20°,20°].
The safety tasks restrict joints ②, ③, and ④ within [22.5°,67.5°], [22.5°,67.5°], and [−10°,10°]
respectively. Letang_2,ang_3,ang_4betheanglesofjoints②,③,④,andthecostisdefinedas:
c =I(ang_2̸∈[22.5°,67.5°], orang_3̸∈[22.5°,67.5°], orang_4̸∈[−10°,10°]). (5)
t
30
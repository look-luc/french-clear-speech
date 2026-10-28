const STIMULI = [
  {
    situation: "base",
    stimulus: "C'est Bon",
  },
  {
    situation: "whisper_in_the_rain",
    stimulus: "c'est le sud à côté du rond bleu",
  },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le trophée sur le triangle vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le whiskey en dessous du rond gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le parrain au-dessus du triangle rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le talon en dessous du triangle vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le thème en dessous du carré rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'allure sur l'étoile grise",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le marquis à côté de l'étoile verte",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est douteux au-dessus du rond vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est relax à côté du triangle rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le débit sur le rond bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la clarté en dessous du cercle vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le recours à côté du carré vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'assaut sur l'étoile verte",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la brebis à côté du rond bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est épars sur le rond vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le godet à côté du triangle gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le micro à côté du rond gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la bougie au-dessus du triangle gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la capsule au-dessus du carré rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la pâleur sur le triangle jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la fureur au-dessus de l'étoile bleue",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le divan en dessous du rond rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le coma en dessous du triangle vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'étape au-dessus du carré gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est amer au-dessus de l'étoile jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le cadet sur le carré rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le savon en dessous de l'étoile jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le bouton à côté de l'étoile bleue",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le ruisseau à côté du triangle rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le bouquin sur le triangle gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le format en dessous du rond gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'ouvrage en dessous du carré gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'aveu en dessous du rond rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est marron en dessous du triangle vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est absurde au-dessus du rond rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la purée en dessous de l'étoile bleue",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'antenne sur l'étoile grise",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le karma sur l'étoile grise",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la flanelle au-dessus du rond bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le sceau sur le rond gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la pancarte au-dessus du triangle gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le cil à côté de l'étoile rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la guitare à côté du carré bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le genou sur le rond gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la moto en dessous de l'étoile bleue",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le menteur à côté du rond vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le gala en dessous du rond rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le cinglé à côté du carré bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la graine en dessous du rond jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le pâté en dessous du triangle bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le précis sur le carré bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'intrus au-dessus du rond jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le secteur en dessous du rond gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le donjon à côté du triangle jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'égard en dessous du triangle rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le poussin en dessous du triangle jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le metteur sur le carré jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'arnaque en dessous de l'étoile rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la charrue sur le triangle jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le cigare au-dessus du rond jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la soucoupe sur l'étoile jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le degré à côté de l'étoile jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'atout sur l'étoile jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le cyprès sur le rond jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le blâme sur le triangle bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la monnaie à côté du rond vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est les patins en dessous du carré vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la fierté au-dessus de l'étoile verte",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le reçu sur le triangle rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le voisin sur le rond bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la rigueur à côté du carré gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le goûter en dessous du carré bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le blafard au-dessus du carré bleu",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est morveux en dessous de l'étoile grise",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le satin sur l'étoile rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le fardeau sur l'étoile bleue",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le talus à côté du rond jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est le bouchon sur le triangle jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'osseux sur le carré gris",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la villa au-dessus de l'étoile jaune",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est cinquante sur le rond vert",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est l'étude sur l'étoile rouge",
  // },
  // {
  //   situation: "whisper_in_the_rain",
  //   stimulus: "c'est la bonté sur le rond vert",
  // },
];

export default STIMULI;

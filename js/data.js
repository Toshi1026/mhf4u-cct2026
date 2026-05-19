/* =============================================================
   data.js  —  ★あなたが編集するのはこのファイルだけ★

   ここを書き換えると、ウェブサイトの中身が変わります。
   js/app.js（仕組み）は触らなくてOKです。

   ◆ 書き方のルール
   - "..." または `...` で囲まれた部分が、あなたが書き込む場所です。
   - 長い文章（context / transformsText / domainRangeText）は
     バッククォート `...` で囲んでいます。改行もそのまま使えます。
   - 空っぽ "" のままにすると、サイト上に「✎ ここに書いてください」
     という枠が表示されます（書き忘れ防止）。
   - 数式は KaTeX 記法。"..." の中では \ を2回（\\）書きます。
       分数 : \\dfrac{上}{下}        例 \\dfrac{1}{x-2}
       2乗 : x^2
       無限大 : \\infty            プラスマイナス : \\pm
       実数の集合 : \\mathbb{R}
       「〜を満たす」: \\mid

   ◆ 重要（採点の注意）
   context / transformsText / domainRangeText の文章は、必ず
   「あなた自身の言葉」で書いてください。AIが書いた文章を貼ると
   最終成績が0点になります。
   ============================================================= */

const SITE = {
  /* ---- トップ（ヒーロー）部分 ---- */
  course:       "MHF4U · Course Culminating Task",
  heroTitle:    "Functions in the Real World",
  heroSubtitle: "A photo album of nine real-world moments, modelled with transformed functions.",
  heroMeta:     "By Toshi · Ken · Taiyo · Justin",

  /* ---- イントロの文章（自分の言葉で。1〜3文くらい） ---- */
  introLead: "While in Malaysia for a year, we realized that mathematics could be found in the simplest of objects – a water bottle, an arm lifted by a friend, and even a slide in the food court. In creating this photo album, we took nine pictures each of actual scenes and modeled them through transformation of the parent functions we learned about this year.",

  footerText: "Functions in the Real World",
};

/* =============================================================
   9つの関数セクション
   順番・関数タイプは課題で決まっています（並べ替えないこと）。
   ============================================================= */

const SECTIONS = [
  {
    /* 1 ===================================================== */
    type:      "Reciprocal Function",
    navLabel:  "1 · 1/x",
    title:     "Arms Acting as a 1/x Function",
    subtitle:  "",
    image:     "images/photo1.jpg",
    imageAlt:  "Arms modelled with a reciprocal (1/x) function",
    desmosId:  "j86chyugkx",
    /* desmosImage: Desmosのスクリーンショット画像。images/ に画像を入れ、
       "images/graph1.jpg" のようにパスを書くと、埋め込みの代わりにその画像が
       「埋め込み風」に表示されます。空 "" なら通常の埋め込みグラフが出ます。 */
    desmosImage: "images/graph1.png",
    parentEq:  "y = \\dfrac{1}{x}",
    modelEq:   "\\begin{gathered}\\text{Right arm:}\\quad y = \\dfrac{2.6}{x+0.4} - 0.3\\\\\\text{Left arm:}\\quad y = \\dfrac{7.9}{x-1.3} + 1.1\\end{gathered}",
    graphTitle:"Vertical displacement as a function of horizontal displacement",
    xLabel:    "Horizontal displacement from zeros (10 cm)",
    yLabel:    "Vertical displacement from zeros (10 cm)",
    domain:    "\\begin{gathered}\\text{Right arm:}\\quad 0 < x < 5.2\\\\\\text{Left arm:}\\quad -4.8 < x < 0\\end{gathered}",
    range:     "\\begin{gathered}\\text{Right arm:}\\quad 0.16 < y < 6.2\\\\\\text{Left arm:}\\quad -4.98 < y < -0.20\\end{gathered}",
    context: "",
    transformsText: `So the parent function is \\(y = \\dfrac{1}{x}\\). I picked it because it has two branches going opposite directions, which is similar to the function. The parent function has a vertical asymptote at \\(x = 0\\) and a horizontal asymptote at \\(y = 0\\), and each transformation I shifted those asymptotes to better align with the actual position of the arms in the photo.

Right arm equation: \\(y = \\dfrac{2.6}{x + 0.4} - 0.3\\). The 2.6 just makes it steep enough to follow the arm angle from shoulder to fingertips. I shifted the asymptote left to \\(x = -0.4\\) because that's where the shoulder is. The -0.3 drops it down a bit since the arm tilts downward when it stretches out.

Left arm: \\(y = \\dfrac{7.9}{x - 1.3} + 1.1\\). Way bigger stretch (7.9) because that arm curves more. Asymptote at \\(x = 1.3\\) puts the steep part near the shoulder on that side. Shifted up 1.1 since that arm sits higher in the photo. I tried doing both arms with one equation first but it's not the same steepness so it didn't work at all. So separate equations make way more sense. This also shows that even though the parent function is symmetric in a sense, the two arms in the photo are not, they have genuinely different curvatures and positions, which is why the parameters a and b, and the shifts, are completely different from each other.`,
    domainRangeText: `Right arm: \\(0 < x < 5.2\\). Basically from next to the body out to where the fingers end, which is roughly 52 cm. Can't include \\(x = -0.4\\) because that's the asymptote. In order to determine the range, I plugged the endpoints of the domain into the equation. When \\(x = 0\\), then \\(y \\approx 6.2\\), while when \\(x = 5.2\\), then \\(y \\approx 0.16\\). Since this particular graph declines on this interval, the range would be \\(0.16 < y < 6.2\\).

Left arm: \\(-4.8 < x < 0\\), so about 48 cm left of center. The asymptote at \\(x = 1.3\\) isn't even in this domain. Plugging in the endpoints, when \\(x = -4.8\\), then \\(y \\approx -0.20\\), while when \\(x = 0\\), then \\(y \\approx -4.98\\). Similarly, since the graph declines on this interval, the range would be \\(-4.98 < y < -0.20\\).`,
  },
  {
    /* 2 ===================================================== */
    type:      "Reciprocal-Squared Function",
    navLabel:  "2 · 1/x²",
    title:     "Arms Acting as a 1/x² Function",
    subtitle:  "",
    image:     "images/photo2.jpg",
    imageAlt:  "Arms modelled with a reciprocal-squared (1/x²) function",
    desmosId:  "ixpqmkwfe7",
    desmosImage: "images/graph2.png",
    parentEq:  "y = \\dfrac{1}{x^2}",
    modelEq:   "\\begin{gathered}\\text{Left arm:}\\quad y = \\dfrac{3.6}{(x-0.9)^2} + 0.5\\\\\\text{Right arm:}\\quad y = \\dfrac{0.7}{(x+0.2)^2} + 0.2\\end{gathered}",
    graphTitle:"Vertical distance as a function of horizontal displacement",
    xLabel:    "Horizontal displacement from zeros (10 cm)",
    yLabel:    "Vertical distance from zeros (10 cm)",
    domain:    "\\begin{gathered}\\text{Left arm:}\\quad -5 < x < 0\\\\\\text{Right arm:}\\quad 0 < x < 5\\end{gathered}",
    range:     "\\begin{gathered}\\text{Left arm:}\\quad 0.60 < y < 4.94\\\\\\text{Right arm:}\\quad 0.23 < y < 17.7\\end{gathered}",
    context: "",
    transformsText: `Parent function \\(y = \\dfrac{1}{x^2}\\). Both branches curve upward which fits better here since both arms are raised. Unlike \\(y = \\dfrac{1}{x}\\) which has one positive and one negative branch, \\(y = \\dfrac{1}{x^2}\\) is always positive, so both branches naturally sit above the x-axis, exactly like how both arms in this photo are lifted above the body.

Left arm: \\(y = \\dfrac{3.6}{(x - 0.9)^2} + 0.5\\). Needed 3.6 to make it tall enough to reach the raised arm height. Shifted right to \\(x = 0.9\\) so the peak sits near the shoulder. +0.5 keeps the bottom from being low.

Right arm: \\(y = \\dfrac{0.7}{(x + 0.2)^2} + 0.2\\). Way smaller stretch (0.7) because this arm is more spread out and gradual, not as steep. The -0.2 and +0.2 were just tiny adjustments to line it up exactly. The difference between 3.6 and 0.7 isn't random, the arms actually look different in the photo. The left arm is bent upward more sharply so it needed a much larger a value to capture that steepness, while the right arm extends more horizontally so a smaller value produces the flatter, wider curve that matches it.`,
    domainRangeText: `Left arm: \\(-5 < x < 0\\), about 50 cm left. Asymptote at \\(x = 0.9\\) is outside this range so the curve is smooth. To determine the range, I used substitution of the endpoint values of the domain. When \\(x = -5\\), \\(y = 0.60\\), while when \\(x = 0\\), \\(y = 4.94\\). Hence, since the graph shows an increasing trend, the range would be \\(0.60 < y < 4.94\\).

Right arm: \\(0 < x < 5\\), same 50 cm to the right. I measured where the arms actually begin and end instead of guessing, which is why the numbers are what they are. Using the same method, when \\(x = 0\\), \\(y = 17.7\\), while when \\(x = 5\\), \\(y = 0.23\\). Therefore, the range will be \\(0.23 < y < 17.7\\).`,
  },
  {
    /* 3 ===================================================== */
    type:      "Polynomial Function (degree > 2)",
    navLabel:  "3 · Polynomial",
    title:     "The Mountain Ridge",
    subtitle:  "Genting Highlands",
    image:     "images/photo3.jpg",
    imageAlt:  "Mountain ridge modelled with a quartic polynomial",
    desmosId:  "khcil48c7s",
    desmosImage: "images/graph3.jpg",
    parentEq:  "f(x) = x^4",
    modelEq:   "y = -0.6(x+4)(x+3.11)^2(x+2.23) + 0.03",
    graphTitle:"Height of the mountain ridge as a function of horizontal position",
    xLabel:    "Horizontal distance of the mountain ridge from forehead (unit)",
    yLabel:    "Height of the mountain ridge from forehead (unit)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid -3.68 \\le x \\le -2.5 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid 0.03 \\le y \\le 0.12 \\,\\}",
    context: `This picture is from a day when me and my friends drive to Genting and take a picture with sunset.`,
    transformsText: `I began with the function of \\(f(x) = x^4\\), and then made it to fit the shape of the mountain ridge in the background of this photo. The equation that I made is \\(y = -0.6(x+4)(x+3.11)^2(x+2.23) + 0.03\\). At first, I wrote the function which is factored form with roots at \\(x = -4\\), \\(x = -3.11\\) with multiplicity 2, and \\(x = -2.23\\). The multiple of two at \\(x = -3.11\\) because the curve does not cross, only touch, which was a good fit with the slight curve in the ridge. The other thing that you will notice about the vertical stretch of -0.6 which is small value and there is squared factor which is \\((x+3.11)^2\\) so graph does not across the \\(x\\)-axis and it touch the \\(x\\) axis and change the direction gradually. The value of +0.03 shifts the entire curve vertical transform upwards and make the graph more matches with the position of the ridge in the photo.`,
    domainRangeText: `I set the domain between \\(-3.68 \\le x \\le -2.5\\) as this is the only part of the curve that really resembles this identifiable mountain ridge. Beyond that part, graph goes to other direction and that will be unmatched with mountain ridge, so I made that limit range in domain. The range of the heights of the ridge above or below the reference point are approximately \\(0.03 \\le y \\le 0.12\\). The values are low because the elevation of the ridge doesn't change much over the section that I modeled.`,
  },
  {
    /* 4 ===================================================== */
    type:      "Trigonometric Function with an Asymptote",
    navLabel:  "4 · Tan / Sec / Csc / Cot",
    title:     "The Red Slide",
    subtitle:  "",
    image:     "images/photo4.jpg",
    imageAlt:  "A red spiral slide modelled with a tangent function",
    desmosId:  "i7exsp2k7z",
    desmosImage: "images/graph4.png",
    parentEq:  "f(x) = \\tan(x)",
    modelEq:   "y = \\tan(0.3x) + 0.9",
    graphTitle:"Height of the slide as a function of horizontal position",
    xLabel:    "Horizontal position along the slide (units)",
    yLabel:    "Height of the slide (units)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid -4 \\le x \\le 4 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid -1.67 \\le y \\le 3.47 \\,\\}",
    context: `The photo is from the Waterfront Residence, where there is the distinct red slide connecting the top floor to the common room. I chose this place since it injects some fun into the daily life of the dorm, and by standing in the foreground, I would capture my own link with this distinctive place. What caught my attention was the way in which the curve is gradual in the middle while steep on either side. The tangent function seemed like the perfect choice for modeling such a curve.`,
    transformsText: `Initially, I used the parent function \\(f(x) = \\tan(x)\\) to represent the slope of the red slide. The most important transformation for me was a horizontal stretch. The coefficient of the function equals \\(0.3x\\). It is not a random figure, since it is based on my estimation of the size of the slide according to the period of the tangent function: \\(\\text{Period} = \\dfrac{\\pi}{|k|}\\). Setting \\(k = 0.3\\) will make the period equal to 10.47. Thus, the vertical asymptotes will move further from the slide, which will allow me to draw a continuous function.

What is more, I introduced a vertical shift to make the tangent curve go up by 0.9 units (\\(c = 0.9\\)). In this way, it became possible to place the inflection point of the curve in the middle of the slide. Moreover, as I did not introduce a vertical stretch (\\(a = 1\\)), it became much easier to match the steepness of the curve.`,
    domainRangeText: `As to the domain, the function of the tangent will be infinitely repeated with multiple vertical asymptotes. However, in my case, it was impossible to keep the function repeating infinitely, since I needed to model only one slide. Therefore, in order to limit the number of repeats of the tangent function, I had to confine the domain to \\([-4, 4]\\) because I needed to model the slide visible on the picture.

Furthermore, as I have confined the domain to the interval which is smaller than the calculated asymptotes (\\(x \\approx \\pm 5.2\\)), I am sure that my function will be continuous enough to describe the true path of sliding.

As to the range, of course, considering the restrictions of the domain, the range of the function will be limited to approximately \\([-1.67, 3.47]\\). As a proof, let us substitute the values of the domain into my function: \\(\\tan(0.3 \\times -4) + 0.9 \\approx -1.67\\) and \\(\\tan(0.3 \\times 4) + 0.9 \\approx 3.47\\). Therefore, this interval corresponds to the real height of the fall from the top to the bottom of the slide.`,
  },
  {
    /* 5 ===================================================== */
    type:      "Sinusoidal Function",
    navLabel:  "5 · Sin / Cos",
    title:     "The Group Wave",
    subtitle:  "",
    image:     "images/photo5.jpg",
    imageAlt:  "Friends posing in a wave, modelled with a sinusoidal function",
    desmosId:  "0a1foymzjl",
    desmosImage: "images/graph5.png",
    parentEq:  "f(x) = \\sin(x)",
    modelEq:   "y = \\sin(-2.2x) + 1.5",
    graphTitle:"Height of the Wave as a function of Horizontal Distance",
    xLabel:    "Horizontal distance across the group (meters)",
    yLabel:    "Height of our linked shoulders (meters)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid -3 \\le x \\le 3 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid 0.5 \\le y \\le 2.5 \\,\\}",
    context: `This picture illustrates one of the memorable moments of being a friend of four people. It shows us linking our hands and posing in various heights to make a continuous wave. This photo reflects all the strong friendships that I have built while staying here. Since the natural position of our shoulders is up and down, I concluded that the best way to model our pose would be a sinusoidal function.`,
    transformsText: `I started with the parent function of \\(f(x) = \\sin(x)\\) to fit the position of our linked shoulders. Firstly, I used the vertical translation of \\(c = 1.5\\) to move our graph upward. The reason for choosing 1.5 as the value is the average distance between the ground and our shoulders in a semi-crouched position. Since I didn't change the amplitude of \\(a = 1\\), the peaks and troughs of the graph precisely corresponded to the maximum and minimum heights of our pose.

Secondly, I used the horizontal compression. The parameter inside the function is \\(-2.2x\\). I did not assume this value by myself; it was obtained using the formula \\(\\text{Period} = \\dfrac{2\\pi}{|k|}\\). Thus, since our wave covered 2.85 units, I calculated that \\(k\\) must be 2.2. Moreover, in order to obtain the reflection across the \\(y\\)-axis, I used the negative sign before \\(k\\). It is necessary because in our case, the person standing on the left hand side of the picture bent his shoulders downward first. Thus, the negative sign makes the wave go in the right direction.`,
    domainRangeText: `As for the domain, since the parent sine function has an unbounded domain, I had to restrict it to \\([-3, 3]\\) in order to accommodate the real-world scenario in the picture. From the grid, the person initiating the wave from the left hand side stands at roughly \\(x = -3\\), while the person on the right hand side ends at \\(x = 3\\). In case I did not restrict the domain, the wave will continue infinitely, suggesting that there are invisible friends who will be doing the wave even outside the picture, which does not make sense.

As for the range, the graph is bounded by the interval \\([0.5, 2.5]\\). Mathematically, this can be seen due to the vertical shift of 1.5 and the amplitude of 1, hence \\(1.5 - 1 = 0.5\\) and \\(1.5 + 1 = 2.5\\). The number 0.5 represents the minimum height where we were crouching down, while 2.5 is the maximum height where the tallest person's shoulders touch. Therefore, all mathematical values lie within the physical dimensions of our bodies.`,
  },
  {
    /* 6 ===================================================== */
    type:      "Exponential Function",
    navLabel:  "6 · Exponential",
    title:     "From Arm to Shoulder",
    subtitle:  "Kuala Lumpur",
    image:     "images/photo6.jpg",
    imageAlt:  "Arm-to-shoulder line modelled with an exponential function",
    desmosId:  "d1rd9szbaq",
    desmosImage: "images/graph6.jpg",
    parentEq:  "f(x) = 18.8^{x}",
    modelEq:   "y = 0.106(18.8)^{x} + 0.194",
    graphTitle:"Height of the arm curve as a function of horizontal position",
    xLabel:    "Horizontal distance arm to shoulder from glass wall (cm)",
    yLabel:    "Height of the shoulder curve from glass wall (cm)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid -1 \\le x \\le 0.8 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid 0.2 \\le y \\le 1.3 \\,\\}",
    context: `This picture is from a day when we shopping and walking around in KL.`,
    transformsText: `At first, I used the simple function which is \\(f(x) = 18.8^x\\) and made transformations to the curve in this photo to model my arm to shoulder. The equation that I made is \\(y = 0.106(18.8)^x + 0.194\\). The base of 18.8 is pretty big and make the graph steeply increase and that was a good fit to the curve from arm to shoulder. The vertical scale of 0.106 makes the curve not steep. The shoulder on the photo doesn't rise to a significant graph unit level. To make the graph fit with model, I did vertical transform +0.194 upward. Both combined brought the exponential curve into a natural place, along the arm to the top of the shoulder.`,
    domainRangeText: `I set the domain between \\(-1 \\le x \\le 0.8\\) because the portion of the curve that corresponds to the visible model of the image. The arm begins from at \\(x = -1\\) and the shoulder's highest point begins from at \\(x = 0.8\\). Therefore, this domain represents exactly the part being modeled. Whereas, the range is from about \\(0.2 \\le y \\le 1.3\\), this range shows that the curve from the arm to the top of the shoulder. The range is relatively compact, even though exponential curve growth steeply because of large number of base which is 18.8. But the domain is short range that the curve does not rise steeply, it includes only the slow to fast growth in the shoulder shape.`,
  },
  {
    /* 7 ===================================================== */
    type:      "Logarithmic Function",
    navLabel:  "7 · Logarithmic",
    title:     "The Water Bottle",
    subtitle:  "",
    image:     "images/photo7.jpg",
    imageAlt:  "Curved top of a water bottle modelled with a logarithmic function",
    desmosId:  "rim1f0rej1",
    desmosImage: "images/graph7.png",
    parentEq:  "f(x) = \\log(x)",
    modelEq:   "f(x) = \\log(x+0.1) + 1",
    graphTitle:"Height of the water bottle as a function of horizontal distance from the back of the bottle",
    xLabel:    "Horizontal distance from the back of the water bottle (units)",
    yLabel:    "Height of the water bottle (units)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid 0 \\le x \\le 1.3 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid 0 \\le y \\le 1.146 \\,\\}",
    context: `I chose this photo because this water bottle is one of the items I often use in my daily life from high school. Since I brought it with me and use it often, it reminds me of my time studying at Sunway.`,
    transformsText: `When I looked at the top part of the water bottle, I noticed that the rounded curve increased quickly at first and then became flatter. This shape looked similar to a logarithmic function, so I selected the parent function \\(y=\\log(x)\\) as my starting point.

At first, the parent function did not match the position of the curve in the photo. The graph started too close to its original vertical asymptote and was not high enough to fit the top of the bottle. To fix this, I applied a horizontal translation of 0.1 units to the left. This changed the function from \\(y=\\log(x)\\) to \\(y=\\log(x+0.1)\\). This helped the curve begin closer to the left side of the rounded shape.

After that, I shifted the graph 1 unit upward so that it lined up better with the height of the bottle in the picture. This resulted in the transformed function \\(y=\\log(x+0.1)+1\\) which closely models the curved top section of the water bottle.

Summary of Transformations:
Parent function — \\(y=\\log(x)\\)
Horizontal translation of 0.1 units to the left — \\(y=\\log(x+0.1)\\)
Vertical translation of 1 unit upward — \\(y=\\log(x+0.1)+1\\)`,
    domainRangeText: `The domain of my function is \\(D: \\{x \\in \\mathbb{R} \\mid 0 \\le x \\le 1.3\\}\\).

I restricted the domain because I only wanted to model the visible curved part at the top of the water bottle. The full logarithmic function continues forever, but the real curve in the photo only appears on a small section of the object. Therefore, I limited the graph to the part that actually matches the picture.

The parent function \\(y=\\log(x)\\) has a domain of \\(D: \\{x \\in \\mathbb{R} \\mid x > 0\\}\\) and a range of \\(R: \\{y \\in \\mathbb{R}\\}\\).

The parent function also has a vertical asymptote at \\(x=0\\) because the logarithmic function is undefined when \\(x \\le 0\\).

After applying the transformations, my function has a vertical asymptote at \\(x=-0.1\\). However, since my restricted domain starts at \\(x=0\\), the asymptote is not included in the modeled section. This means the graph is continuous on the restricted interval.

To find the range, I evaluated the function at the endpoints of the domain. When \\(x=0\\) the output is approximately \\(y=0\\), and when \\(x=1.3\\) the output is approximately \\(y=1.146\\). Therefore, the range of my function is approximately \\(R: \\{y \\in \\mathbb{R} \\mid 0 \\le y \\le 1.146\\}\\).`,
  },
  {
    /* 8 — 自由選択 ========================================== */
    type:      "Absolute Value Function",
    navLabel:  "8 · Absolute Value",
    title:     "The V-Neck Shirt",
    subtitle:  "",
    image:     "images/photo8.jpg",
    imageAlt:  "V-shape of a T-shirt modelled with an absolute value function",
    desmosId:  "weeukqkmnf",
    desmosImage: "images/graph8.png",
    parentEq:  "f(x) = |x|",
    modelEq:   "f(x) = |x|",
    graphTitle:"Height of the logo as a function of horizontal distance from the logo",
    xLabel:    "Horizontal distance from my T-shirt logo (units)",
    yLabel:    "Height of the logo (units)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid -0.93 < x < 0.56 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid 0 \\le y < 0.93 \\,\\}",
    context: `I chose this photo because this T-shirt is very meaningful to me. Before I came to study abroad, my parents bought matching T-shirts for my family in different colors. Because of that, this shirt reminds me of my family and my life before coming to Malaysia.`,
    transformsText: `When I looked at the logo design on the shirt, I noticed that part of the pattern made a clear V-shape. This shape closely resembled an absolute value function, so I selected the parent function \\(y=|x|\\) as my starting point.

At first, I compared the parent function with the V-shape in the logo. The original absolute value function already matched the direction and basic shape of the logo because both sides of the graph rise from a lowest point. Since the vertex of the graph was already positioned at the lowest point of the V-shape, I did not need to apply any horizontal or vertical translations.

I also did not apply a vertical stretch or compression because the steepness of the parent function matched the V-shape well enough in the photo. Therefore, the final function \\(y=|x|\\) remained.

I only restricted the domain so that the graph showed the section of the logo that was visible and useful for my model.

Summary of Transformations:
Parent function — \\(y=|x|\\)
No horizontal translation — \\(y=|x|\\)
No vertical translation — \\(y=|x|\\)
No vertical stretch or compression — \\(y=|x|\\)
Final function — \\(y=|x|\\)`,
    domainRangeText: `The domain of my function is \\(D: \\{x \\in \\mathbb{R} \\mid -0.93 < x < 0.56\\}\\).

I restricted the domain because I only wanted to model the visible V-shape in the T-shirt logo. The full absolute value function continues forever in both directions, but the real logo only shows a small part of the V-shape. Therefore, I limited the graph to the section that matches the actual design in the photo.

The parent function \\(y=|x|\\) has a domain of \\(D: \\{x \\in \\mathbb{R}\\}\\) and a range of \\(R: \\{y \\in \\mathbb{R} \\mid y \\ge 0\\}\\). This means the function can take any real x-value, but the output is always zero or positive.

For my restricted model, the lowest point is at the vertex, \\((0,0)\\), so the minimum y-value is 0. The highest visible part of the modeled section is close to \\(y=0.93\\). Therefore, the range of my function is approximately \\(R: \\{y \\in \\mathbb{R} \\mid 0 \\le y < 0.93\\}\\).`,
  },
  {
    /* 9 — 自由選択 ========================================== */
    type:      "Quadratic Function",
    navLabel:  "9 · Quadratic",
    title:     "Family Mart's Bottle's Curve",
    subtitle:  "",
    image:     "images/photo9.jpg",
    imageAlt:  "Curved part of a water bottle modelled with a quadratic function",
    desmosId:  "fczaoqqmz0",
    desmosImage: "images/graph9.png",
    parentEq:  "f(x) = x^2",
    modelEq:   "f(x) = -2.3x^2 + 0.1",
    graphTitle:"Height of the water bottle as a function of horizontal distance across the top",
    xLabel:    "Horizontal distance across the top of the water bottle (units)",
    yLabel:    "Height of the water bottle (units)",
    domain:    "\\{\\, x \\in \\mathbb{R} \\mid -0.9 \\le x \\le 0.9 \\,\\}",
    range:     "\\{\\, y \\in \\mathbb{R} \\mid -1.763 \\le y \\le 0.1 \\,\\}",
    context: `I chose this photo because this water bottle is very meaningful to me. I have used this type of water bottle since I was in elementary school, and this specific bottle has been with me since Grade 10. I brought it to Malaysia because it is one of my favourite items and it reminds me of my daily life as a student.`,
    transformsText: `When I looked at the top part of the bottle, I noticed that the rounded shape looked similar to a downward-opening parabola. Therefore, I selected the parent function \\(y=x^2\\) as my starting point.

At first, the parent function opened upward, but the curve on the top of the water bottle opened downward. Because of this, I reflected the graph over the x-axis. This changed the direction of the parabola so that it could match the arch shape in the photo.

After that, I applied a vertical stretch by a factor of 2.3. This made the parabola narrower and steeper, which helped it fit the curved top of the bottle more accurately. Finally, I shifted the graph 0.1 units upward so that the highest point of the graph lined up with the top part of the bottle. This resulted in the function \\(y=-2.3x^2+0.1\\), which closely models the curved shape in the photograph.

Summary of Transformations:
Parent function — \\(y=x^2\\)
Reflection across the x-axis — \\(y=-x^2\\)
Vertical stretch by a factor of 2.3 — \\(y=-2.3x^2\\)
Vertical translation of 0.1 units upward — \\(y=-2.3x^2+0.1\\)
Final function — \\(y=-2.3x^2+0.1\\)`,
    domainRangeText: `The domain of my function is \\(D: \\{x \\in \\mathbb{R} \\mid -0.9 \\le x \\le 0.9\\}\\).

I restricted the domain because I only wanted to model the visible curved section at the top of the water bottle. The full quadratic function would continue forever to the left and right, but the real curve in the photo only exists on the top part of the bottle. Therefore, I limited the domain to the section that matches the actual shape in the image.

The parent function \\(y=x^2\\) has a domain of \\(D: \\{x \\in \\mathbb{R}\\}\\) and a range of \\(R: \\{y \\in \\mathbb{R} \\mid y \\ge 0\\}\\).

However, after the reflection and vertical stretch, my function opens downward. The highest point of the graph is the vertex, \\((0, 0.1)\\). This means the maximum value of the function is \\(y=0.1\\).

Since the function is restricted from \\(x=-0.9\\) to \\(x=0.9\\), I found the lowest output by checking the endpoints. At both endpoints, the y-value is approximately \\(y=-1.763\\). Therefore, the range of my function is approximately \\(R: \\{y \\in \\mathbb{R} \\mid -1.763 \\le y \\le 0.1\\}\\).`,
  },
];

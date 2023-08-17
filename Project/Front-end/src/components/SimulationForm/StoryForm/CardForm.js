// CardForm.js
import React, { useState, useEffect } from 'react';
import { useSprings, animated } from '@react-spring/web';
import { useDrag } from 'react-use-gesture';

import styles from './CardForm.module.css';
import '../../../components/App/App.css';
import SelectForm from '../SelectForm/SelectForm';
import NextBtn from '../NextBtn';

const to = (i) => ({
  x: 0,
  y: i * -4,
  scale: 1,
  rot: -10 + Math.random() * 20,
  delay: i * 100,
});

const cardsFirst = [
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card1.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card2.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card3.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card4.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card5.png',
];

const cardsFinal = [
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/drug5.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/drug4.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/drug3.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/drug2.png',
  'https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/drug.png',
];

const from = () => ({ x: 0, rot: 0, scale: 1.5, y: -1000 });

const trans = (r, s) => `perspective(1500px) rotateX(30deg) rotateY(${r / 10}deg) rotateZ(${r}deg) scale(${s})`;

function Deck({ onAllCardsGone, flag, onNext }) {
  const [gone] = useState(() => new Set());

  const cards = flag === 1 ? cardsFirst : cardsFinal;

  const [props, api] = useSprings(cards.length, (i) => ({
    ...to(i),
    from: from(i),
  }));

  const bind = useDrag(({ args: [index], down, movement: [mx], direction: [xDir], velocity }) => {
    const trigger = velocity > 0.2;
    const dir = xDir < 0 ? -1 : 1;
    if (!down && trigger) gone.add(index);
    api.start((i) => {
      if (index !== i) return;
      const isGone = gone.has(index);
      const x = isGone ? (200 + window.innerWidth) * dir : down ? mx : 0;
      const rot = mx / 100 + (isGone ? dir * 10 * velocity : 0);
      const scale = down ? 1.1 : 1;
      return {
        x,
        rot,
        scale,
        delay: undefined,
        config: { friction: 50, tension: down ? 800 : isGone ? 200 : 500 },
      };
    });
    if (!down && gone.size === cards.length) {
      setTimeout(() => {
        gone.clear();
        api.start((i) => to(i));

        if (flag === 2) {
          onNext();
        } else {
          onAllCardsGone();
        }
      }, 100);
    }
  });

  return (
    <>
      {props.map(({ x, y, rot, scale }, i) => (
        <animated.div className={styles.deck} key={i} style={{ x, y }}>
          <animated.div
            {...bind(i)}
            style={{
              transform: trans(rot, scale),
              backgroundImage: `url(${cards[i]})`,
            }}
          ></animated.div>
        </animated.div>
      ))}
    </>
  );
}

export default function CardForm({ onNext, initialFlag }) {
  const [showSelectForm, setShowSelectForm] = useState(false);
  const [flag, setFlag] = useState(initialFlag);

  useEffect(() => {
    setFlag(initialFlag);
  }, [initialFlag]);

  const handleAllCardsGone = () => {
    setShowSelectForm(true);
    if (flag === 1) {
      setFlag(2);
    }
  };

  return (
    <div className={styles.container}>
      {showSelectForm ? (
        <SelectForm />
      ) : (
        <div className={styles.container}>
          <div className={styles.deckTitle}>도파민 카드</div>
          <Deck onAllCardsGone={handleAllCardsGone} flag={flag} onNext={onNext} />
          <img
            src="https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/fingerGesture.gif"
            alt="Finger Gesture"
            className={styles.fingerGesture}
          />
        </div>
      )}
    </div>
  );
}

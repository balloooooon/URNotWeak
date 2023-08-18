import React, { Component } from 'react';
import * as THREE from 'three';

class UserInteraction extends Component {
  constructor(props) {
    super(props);
    this.state = {
      touchCount: 0,
    };

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.cube = null;
  }

  componentDidMount() {
    document.body.appendChild(this.renderer.domElement);

    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    this.cube = new THREE.Mesh(geometry, material);
    this.scene.add(this.cube);

    this.camera.position.z = 5;

    this.animate();
  }

  handleTouch = () => {
    this.setState(prevState => ({
      touchCount: prevState.touchCount + 1,
    }));

    if (this.cube) {
      this.cube.material.color.setHex(this.state.touchCount % 2 === 0 ? 0x00ff00 : 0xff0000);
    }
  };

  animate = () => {
    requestAnimationFrame(this.animate);
    this.renderer.render(this.scene, this.camera);
  };

  render() {
    return (
      <div onTouchStart={this.handleTouch}>
        {this.state.touchCount} touches
      </div>
    );
  }
}

export default UserInteraction;

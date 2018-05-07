import React from 'react';


class Network extends React.Component {

    drawNetwork(){
        var g = {
            nodes: this.props.network['nodes'],
            edges: this.props.network['edges'],
        };

        var s = new sigma({
            graph: g,
            container: 'network',
            settings: {
                drawLabels: false,
            },
            renderer: {
                container: document.getElementById('network'),
                type: 'canvas',
            },
        });
    }

    clearNetwork(){
        $(this.refs.network).empty();
    }

    componentDidMount(){
        this.drawNetwork();
    }

    componentWillUnmount(){
        this.clearNetwork();
    }

    render(){
        return <div
            style={{height:'100%', width:"100%", border: '1px solid black'}}
            ref='network' id='network'>
        </div>;
    }
}

Network.propTypes = {
    network: React.PropTypes.object.isRequired,
};

export default Network;

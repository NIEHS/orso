import React from 'react';
import PieChart from './PieChart';


class SmallUserView extends React.Component {

    constructor(props) {
        super(props);
        this.state = {is_favorite: (props.meta_data['is_favorite'] === 'true')};
    }

    componentDidMount(){

        let add_favorite_url = this.props.urls['add_favorite'],
            remove_favorite_url = this.props.urls['remove_favorite'],
            hide_recommendation_url = this.props.urls['hide_recommendation'];
        let self = this;

        $(this.refs.favorite_button).on('click', function () {
            if (self.state.is_favorite) {
                self.setState({
                    is_favorite: false
                });

                let favorite_count = parseInt($('#favorite_counts').html());
                favorite_count = favorite_count - 1;
                $('#favorite_counts').html(favorite_count);

                $.ajax({url: remove_favorite_url});

            } else {
                self.setState({
                    is_favorite: true
                });

                let favorite_count = parseInt($('#favorite_counts').html());
                favorite_count = favorite_count + 1;
                $('#favorite_counts').html(favorite_count);

                $.ajax({url: add_favorite_url});
            }
        });

        $(this.refs.remove_recommendation_button).on('click', function () {
            let recommended_count = parseInt($('#recommended_counts').html());
            recommended_count = recommended_count - 1;
            $('#recommended_counts').html(recommended_count);

            $.ajax({url: hide_recommendation_url});
        });

        $(this.refs.remove_favorite_button).on('click', function () {
            let favorite_count = parseInt($('#favorite_counts').html());
            favorite_count = favorite_count - 1;
            $('#favorite_counts').html(favorite_count);

            $.ajax({url: remove_favorite_url});
        });
    }

    render(){
        var id_select = 'panel_' + this.props.meta_data['id'];
        var id_css_select = '#' + id_select;

        var pie_id_1 = '' + this.props.meta_data['pk'] + '_0';
        var pie_id_2 = '' + this.props.meta_data['pk'] + '_1';

        return <div className="panel panel-default" id={id_select}>
            <div className="panel-heading">
                <div className="panel-title pull-left">
                    <a href={this.props.urls['detail']}>{this.props.meta_data['username']}</a>
                </div>
                <div className="panel-title pull-right">
                    {this.props.display_favorite &&
                        <button type="button" ref="favorite_button" className="panel-close-button">
                            &nbsp;
                            {this.state.is_favorite ? (
                                <span className="glyphicon glyphicon-star"></span>
                            ) : (
                                <span className="glyphicon glyphicon-star-empty"></span>
                            )}
                        </button>
                    }
                    {this.props.display_remove_recommendation &&
                        <button type="button" ref="remove_recommendation_button" className="panel-close-button"
                                data-target={id_css_select} data-dismiss="alert">
                            &nbsp;<span className="glyphicon glyphicon-remove"></span>
                        </button>
                    }
                    {this.props.display_remove_favorite &&
                        <button type="button" ref="remove_favorite_button" className="panel-close-button"
                                data-target={id_css_select} data-dismiss="alert">
                            &nbsp;<span className="glyphicon glyphicon-remove"></span>
                        </button>
                    }
                </div>
                <div className="clearfix"></div>
            </div>
            <div className="panel-body">
                <div className="row">
                    <div style={{height:"200px"}} className="col-sm-6">
                        <ul>
                            <li>{this.props.meta_data['dataset_number']} datasets</li>
                            <li>Following {this.props.meta_data['user_following_number']} users</li>
                            <li>Followed by {this.props.meta_data['user_followed_by_number']} users</li>
                            <li>{this.props.meta_data['data_favorite_number']} favorite datasets</li>
                            <li>Datasets favorited {this.props.meta_data['data_favorited_by_number']} times</li>
                        </ul>
                    </div>
                    {Object.keys(this.props.plot_data['assembly_counts']).length > 0 &&
                        <div className="col-sm-3">
                            <h4 style={{textAlign:"center"}}>Assembly</h4>
                            <PieChart
                                data={this.props.plot_data['assembly_counts']}
                                id={pie_id_1}
                            />
                        </div>
                    }
                    {Object.keys(this.props.plot_data['experiment_counts']).length > 0 &&
                        <div className="col-sm-3">
                            <h4 style={{textAlign:"center"}}>Experiment types</h4>
                            <PieChart
                                data={this.props.plot_data['experiment_counts']}
                                id={pie_id_2}
                            />
                        </div>
                    }
                </div>
            </div>
        </div>;

    }
}

SmallUserView.propTypes = {
    meta_data: React.PropTypes.object.isRequired,
    plot_data: React.PropTypes.object.isRequired,
    urls: React.PropTypes.object.isRequired,

    display_favorite: React.PropTypes.bool,
    display_remove_recommendation: React.PropTypes.bool,
    display_remove_favorite: React.PropTypes.bool,
};

SmallUserView.defaultProps = {
    display_favorite: false,
    display_remove_recommendation: false,
    display_remove_favorite: false,
};

export default SmallUserView;

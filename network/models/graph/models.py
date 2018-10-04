from neomodel import (
    FloatProperty,
    IntegerProperty,
    RelationshipFrom,
    RelationshipTo,
    StructuredNode,
    StructuredRel,
    StringProperty,
)


# RELATIONSHIPS
class RelSimilarity(StructuredRel):
    score = FloatProperty(required=True)


# NODES
class GraphUser(StructuredNode):
    rel_id = IntegerProperty(unique_index=True, required=True)

    # PROPERTIES
    username = StringProperty(required=True)

    # RELATIONSHIPS
    owned_experiment = RelationshipTo('GraphExperiment', 'OWN')
    favorited_experiment = RelationshipTo(
        'GraphExperiment', 'FAVORITE')
    recommended_experiment = RelationshipFrom(
        'GraphExperiment', 'RECOMMENDATION')

    followed_user = RelationshipTo(
        'GraphUser', 'FOLLOW')
    following_user = RelationshipFrom(
        'GraphUser', 'FOLLOW')


class GraphExperiment(StructuredNode):
    rel_id = IntegerProperty(unique_index=True, required=True)

    # PROPERTIES
    name = StringProperty(required=True)

    experiment_type_id = IntegerProperty(required=True)
    organism_id = IntegerProperty(required=True)

    # RELATIONSHIPS
    dataset = RelationshipFrom('GraphDataset', 'OF')

    sim_to_experiment = RelationshipTo(
        'GraphExperiment', 'SIMILARITY', model=RelSimilarity)
    sim_from_experiment = RelationshipFrom(
        'GraphExperiment', 'SIMILARITY', model=RelSimilarity)

    owning_user = RelationshipFrom('GraphUser', 'OWN')
    recommended_user = RelationshipTo(
        'GraphUser', 'RECOMMENDATION')
    favoriting_user = RelationshipFrom(
        'GraphUser', 'FAVORITE')


class GraphDataset(StructuredNode):
    rel_id = IntegerProperty(unique_index=True, required=True)

    # PROPERTIES
    name = StringProperty(required=True)

    assembly_id = IntegerProperty(required=True)

    # RELATIONSHIPS
    sim_to_dataset = RelationshipTo(
        'GraphDataset', 'SIMILARITY', model=RelSimilarity)
    sim_from_dataset = RelationshipFrom(
        'GraphDataset', 'SIMILARITY', model=RelSimilarity)

    experiment = RelationshipTo('GraphExperiment', 'OF')

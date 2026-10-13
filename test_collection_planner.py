
import unittest

from collection_planner import create_collection_plan


class TestCollectionPlanner(unittest.TestCase):

    def test_hinge_plan_has_six_roles(self):
        plan = create_collection_plan(
            "Find the best 6 photos for Hinge"
        )

        self.assertEqual(plan.goal, "hinge")
        self.assertEqual(plan.count, 6)

    def test_roles_are_unique(self):
        plan = create_collection_plan("Hinge")

        names = [role.name for role in plan.roles]

        self.assertEqual(len(names), len(set(names)))

    def test_roles_have_search_queries(self):
        plan = create_collection_plan("Hinge")

        for role in plan.roles:
            self.assertTrue(role.query.strip())
            self.assertTrue(role.description.strip())

    def test_case_insensitive_goal(self):
        plan = create_collection_plan("HINGE profile photos")

        self.assertEqual(plan.goal, "hinge")

    def test_empty_goal_rejected(self):
        with self.assertRaises(ValueError):
            create_collection_plan("")

    def test_unsupported_goal_rejected(self):
        with self.assertRaises(ValueError):
            create_collection_plan("LinkedIn profile")

    def test_unsupported_count_rejected(self):
        with self.assertRaises(ValueError):
            create_collection_plan("Hinge", count=5)


if __name__ == "__main__":
    unittest.main()

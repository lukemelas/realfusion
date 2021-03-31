import { Flex, Heading, Stack, HStack, SimpleGrid, Wrap, Center } from '@chakra-ui/react'
import { Container } from 'next/app'

const institutions = {
  1: "Oxford University",
  2: "Harvard University",
}

const authors = [
  {
    'name': 'Luke Melas-Kyraizi',
    'institutions': [1, 2]
  },
]

export const Hero = ({ title }: { title: string }) => (
  <Container>
    <Heading fontSize="calc(20px + 0.5vw)" pt="5vh" pb="1vh">{title}</Heading>
    <Wrap justify="center">

      {
        authors.map((author) =>
          <span>
            {author.name}
            <sup> {author.institutions.toString()}</sup>
          </span>
        )
      }
    </Wrap>
    <Wrap justify="center">
      {
        Object.entries(institutions).map(tuple =>
          <span>
            <sup>{tuple[0]}  </sup>
            {tuple[1]}
          </span>
        )
      }
    </Wrap>
    <Heading fontSize="calc(16px + 0.5vw)">Authors</Heading>
  </Container>
)

Hero.defaultProps = {
  title: 'Academic Project Template',
}